from ContextTimer import ContextTimer

from typing import Dict, List, Tuple, Iterable
import bitarray
import chess
import chess.pgn
import io

from utils import format_fullmove, to_int, bits_required_for_n_states

BB_DOUBLE_PAWN_MOVE_RANKS = chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_7 | chess.BB_RANK_8
# noinspection PyProtectedMember
BB_PAWN_MOVES = [[chess._step_attacks(sq, deltas[0] if chess.BB_SQUARES[sq] & BB_DOUBLE_PAWN_MOVE_RANKS else deltas[1])
                  for sq in chess.SQUARES]
                 for deltas in (((-7, -8, -9, -16), (-7, -8, -9)), ((7, 8, 9, 16), (7, 8, 9)))]
# noinspection PyProtectedMember
BB_KNIGHT_MOVES = chess.BB_KNIGHT_ATTACKS
# noinspection PyProtectedMember
BB_BISHOP_MOVES = [chess._sliding_attacks(sq, 0, (-9, -7, 7, 9)) for sq in chess.SQUARES]
# noinspection PyProtectedMember
BB_ROOK_MOVES = [chess._sliding_attacks(sq, 0, (-8, -1, 1, 8)) for sq in chess.SQUARES]
BB_QUEEN_MOVES = [BB_ROOK_MOVES[sq] | BB_BISHOP_MOVES[sq] for sq in chess.SQUARES]
BB_KING_MOVES = chess.BB_KING_ATTACKS
# Manually add castling moves, TODO support 960 castling to_squares?
BB_KING_MOVES[chess.E1] |= chess.BB_SQUARES[chess.G1] | chess.BB_SQUARES[chess.C1]
BB_KING_MOVES[chess.E8] |= chess.BB_SQUARES[chess.G8] | chess.BB_SQUARES[chess.C8]

BB_PIECE_MOVES: Dict[chess.PieceType, List] = {
    chess.PAWN: BB_PAWN_MOVES,
    chess.KNIGHT: BB_KNIGHT_MOVES,
    chess.BISHOP: BB_BISHOP_MOVES,
    chess.ROOK: BB_ROOK_MOVES,
    chess.QUEEN: BB_QUEEN_MOVES,
    chess.KING: BB_KING_MOVES
}

PROMOTION_PIECE_TYPE_BB_MAP = {
    chess.QUEEN: bitarray.bitarray('00'),
    chess.ROOK: bitarray.bitarray('01'),
    chess.BISHOP: bitarray.bitarray('10'),
    chess.KNIGHT: bitarray.bitarray('11'),
}
REVERSE_PROMOTION_PIECE_TYPE_BB_MAP = {
    v.to01(): k for k, v in PROMOTION_PIECE_TYPE_BB_MAP.items()
}
BB_PRE_PROMOTION_RANKS = chess.BB_RANK_2 | chess.BB_RANK_7
BB_PROMOTION_RANKS = chess.BB_RANK_1 | chess.BB_RANK_8


def _get_movable_pieces_bb(board: chess.Board,
                           *,
                           mask_legal: bool = False,
                           mask_pseudo_legal: bool = False) -> chess.Bitboard:
    """
    If ``mask_legal`` or ``mask_pseudo_legal``, return a bitboard with all ``from_square``s of the
    legal/pseudo-legal moves in ``board``. Otherwise, return ``board.occupied_co[board.turn]``.
    """
    occupied = board.occupied_co[board.turn]

    if mask_legal:
        occupied_movable = 0
        for m in board.generate_legal_moves():
            occupied_movable |= chess.BB_SQUARES[m.from_square]
            if occupied_movable == occupied:
                # Stop once we got every possible from_square
                break
        return occupied_movable

    if mask_pseudo_legal:
        occupied_movable = 0
        for m in board.generate_pseudo_legal_moves():
            occupied_movable |= chess.BB_SQUARES[m.from_square]
            if occupied_movable == occupied:
                # Stop once we got every possible from_square
                break
        return occupied_movable

    return occupied


def _get_actions_bb(from_square: chess.Square,
                    board: chess.Board,
                    *,
                    mask_legal: bool = False,
                    mask_pseudo_legal: bool = False) -> chess.Bitboard:
    """
    If ``mask_legal`` or ``mask_pseudo_legal``, return a bitboard with all ``to_square``s of the
    legal/pseudo-legal moves in ``board``. Otherwise, return a pre-generated bitboard with all squares
    the piece on ``from_square`` could possibly move or capture to.
    """
    if mask_legal:
        piece_moves = 0
        for m in board.generate_legal_moves(from_mask=chess.BB_SQUARES[from_square]):
            piece_moves |= chess.BB_SQUARES[m.to_square]
        return piece_moves

    if mask_pseudo_legal:
        piece_moves = 0
        for m in board.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[from_square]):
            piece_moves |= chess.BB_SQUARES[m.to_square]
        return piece_moves

    # Return pre-generated bit masks, which assume an empty board, and pawns can capture
    piece = board.piece_at(from_square)
    if piece.piece_type == chess.PAWN:
        return BB_PIECE_MOVES[piece.piece_type][piece.color][from_square]
    return BB_PIECE_MOVES[piece.piece_type][from_square]


def _encode_occupied_idx(s: chess.Square,
                         occupied: chess.Bitboard) -> int:
    """
    Take a square and return the number of truthy bits that come
    before it in the ``occupied`` bitboard for the side to move.
    """
    assert chess.BB_SQUARES[s] & occupied, 'Expected ``s`` to be an occupied square'
    return occupied.bit_count() - (occupied >> s).bit_count()


def _decode_occupied_idx(index: int,
                         occupied: chess.Bitboard) -> chess.Square:
    """
    Take the number of truthy bits that come before the next square in ``occupied`` and return that square's index.

    Thanks, ChatGPT!
    """
    # Initialize variables
    square = None  # Represents an invalid square

    # Iterate through the bits of the occupied bitboard
    while occupied:
        # Find the least significant set bit
        lsb = occupied & -occupied

        # Decrement the index and check if it matches the target index
        if index == 0:
            square = chess.lsb(lsb)
            break

        # Clear the least significant set bit
        occupied ^= lsb

        # Decrement the index
        index -= 1

    assert square is not None
    return square


def encode_from_square(move: chess.Move,
                       board: chess.Board,
                       *,
                       mask_legal: bool = False,
                       mask_pseudo_legal: bool = False) -> bitarray.bitarray:
    """
    Encode ``move.from_square`` either by indexing it against the truthy bits in ``board.occupied_co[board.turn]``,
    or by indexing it against the union of all ``from_square``s of all legal/pseudo-legal moves.
    """
    movable_pieces_bb = _get_movable_pieces_bb(board,
                                               mask_legal=mask_legal,
                                               mask_pseudo_legal=mask_pseudo_legal)
    assert movable_pieces_bb
    idx = _encode_occupied_idx(move.from_square, movable_pieces_bb)
    num_bits = bits_required_for_n_states(movable_pieces_bb.bit_count())
    if num_bits == 0:
        return bitarray.bitarray()
    return bitarray.bitarray(format(idx, f'0{num_bits}b'))


def decode_from_square(from_square_bits: bitarray.bitarray,
                       board: chess.Board,
                       *,
                       mask_legal: bool = False,
                       mask_pseudo_legal: bool = False) -> chess.Square:
    """
    Decode ``from_square_bits`` either by indexing it against the truthy bits in ``board.occupied_co[board.turn]``,
    or by indexing it against the union of all ``from_square``s of all legal/pseudo-legal moves.
    """
    movable_pieces_bb = _get_movable_pieces_bb(board,
                                               mask_legal=mask_legal,
                                               mask_pseudo_legal=mask_pseudo_legal)
    return _decode_occupied_idx(to_int(from_square_bits), movable_pieces_bb)


def encode_to_square(move: chess.Move,
                     board: chess.Board,
                     *,
                     mask_legal: bool = False,
                     mask_pseudo_legal: bool = False) -> bitarray.bitarray:
    """
    Encode ``move.to_square`` either by indexing it against a pre-generated bitboard of all potential legal moves
    for the piece at ``move.from_square``, or by indexing it against the union of all ``to_square``s of all
    legal/pseudo-legal moves.
    """
    actions_bb = _get_actions_bb(move.from_square,
                                 board,
                                 mask_legal=mask_legal,
                                 mask_pseudo_legal=mask_pseudo_legal)
    idx = _encode_occupied_idx(move.to_square, actions_bb)
    num_bits = bits_required_for_n_states(actions_bb.bit_count())
    if num_bits == 0:
        return bitarray.bitarray()
    return bitarray.bitarray(format(idx, f'0{num_bits}b'))


def decode_to_square(from_square: chess.Square,
                     to_square_bits: bitarray.bitarray,
                     board: chess.Board,
                     *,
                     mask_legal: bool = False,
                     mask_pseudo_legal: bool = False) -> chess.Square:
    """
    Decode ``to_square_bits`` either by indexing it against a pre-generated bitboard of all potential legal moves
    for the piece at ``from_square``, or by indexing it against the union of all ``to_square``s of all
    legal/pseudo-legal moves.
    """
    actions_bb = _get_actions_bb(from_square,
                                 board,
                                 mask_legal=mask_legal,
                                 mask_pseudo_legal=mask_pseudo_legal)
    assert actions_bb
    return _decode_occupied_idx(to_int(to_square_bits), actions_bb)


def encode_promotion_piece_type(piece_type: chess.PieceType) -> bitarray.bitarray:
    """ Encode ``piece_type`` into a bitarray representation with 2 bits. """
    return PROMOTION_PIECE_TYPE_BB_MAP[piece_type]


def decode_promotion_piece_type(bits: bitarray.bitarray) -> Tuple[chess.PieceType, int]:
    """ Decode the first 2 bits of ``bits`` into a ``chess.PieceType``. """
    # TODO micro-optimize (loop through k, v in PROMOTION_PIECE_TYPE_BB_MAP.items() and check if bits == v ?)
    return REVERSE_PROMOTION_PIECE_TYPE_BB_MAP[bits[:2].to01()], 2


def encode_move(move: chess.Move,
                board: chess.Board,
                *,
                mask_legal: bool = False,
                mask_pseudo_legal: bool = False,
                debug: bool = False) -> bitarray.bitarray:
    """
    Encode ``move`` into a bitarray representation by concatenating the encodings of its ``from_square``
    and ``to_square``. See ``encode_from_square()`` and ``encode_to_square()``.
    """
    encoded_from_sq = encode_from_square(move,
                                         board,
                                         mask_legal=mask_legal,
                                         mask_pseudo_legal=mask_pseudo_legal)
    encoded_to_sq = encode_to_square(move,
                                     board,
                                     mask_legal=mask_legal,
                                     mask_pseudo_legal=mask_pseudo_legal)

    if debug:
        print(f'{encoded_from_sq.to01()}->{encoded_to_sq.to01()}', end='')

    bits = encoded_from_sq + encoded_to_sq

    if move.promotion:
        promotion_bits = encode_promotion_piece_type(move.promotion)
        bits += promotion_bits

        if debug:
            print(f'={promotion_bits.to01()}', end='')

    if debug:
        print(f' ({move.uci()})', end='')

    return bits


def decode_move(bits: bitarray.bitarray,
                board: chess.Board,
                *,
                mask_legal: bool = False,
                mask_pseudo_legal: bool = False,
                debug: bool = False) -> Tuple[chess.Move, int]:
    """
    Decode the first bits in ``bits`` into a move by decoding ``from_square`` and ``to_square`` based on
    ``board``. Return the decoded move and the number of bits consumed to decode it. See ``decode_from_square()``
    and ``decode_to_square()``.
    """

    total_consumed = 0

    # From square
    movable_pieces_bb = _get_movable_pieces_bb(board,
                                               mask_legal=mask_legal,
                                               mask_pseudo_legal=mask_pseudo_legal)
    from_square_bit_len = bits_required_for_n_states(movable_pieces_bb.bit_count())
    from_square = decode_from_square(bits[:from_square_bit_len],
                                     board,
                                     mask_legal=mask_legal,
                                     mask_pseudo_legal=mask_pseudo_legal)
    if debug:
        print(f'{bits[:from_square_bit_len].to01()}->', end='')
    bits = bits[from_square_bit_len:]
    total_consumed += from_square_bit_len

    # To square
    actions_bb = _get_actions_bb(from_square,
                                 board,
                                 mask_legal=mask_legal,
                                 mask_pseudo_legal=mask_pseudo_legal)
    to_square_bit_len = bits_required_for_n_states(actions_bb.bit_count())
    to_square = decode_to_square(from_square,
                                 bits[:to_square_bit_len],
                                 board,
                                 mask_legal=mask_legal,
                                 mask_pseudo_legal=mask_pseudo_legal)
    if debug:
        print(f'{bits[:to_square_bit_len].to01()}', end='')
    bits = bits[to_square_bit_len:]
    total_consumed += to_square_bit_len

    move = chess.Move(from_square, to_square)

    # Promotion
    if board.piece_type_at(from_square) == chess.PAWN \
            and chess.BB_SQUARES[from_square] & BB_PRE_PROMOTION_RANKS \
            and chess.BB_SQUARES[to_square] & BB_PROMOTION_RANKS:
        promotion_piece_type, promotion_consumed = decode_promotion_piece_type(bits[:2])
        move.promotion = promotion_piece_type
        if debug:
            print(f'={bits[:promotion_consumed].to01()}', end='')
        bits = bits[promotion_consumed:]
        total_consumed += promotion_consumed

    if debug:
        print(f' ({move.uci()})', end='')

    return move, total_consumed


def encode_moves(moves: Iterable[chess.Move],
                 starting_fen: str,
                 *,
                 mask_legal: bool = False,
                 mask_pseudo_legal: bool = False,
                 debug: bool = False) -> bitarray.bitarray:
    """ Return a concatenation of each encoded move in ``moves``. See ``encode_move()``. """
    encoded_moves = bitarray.bitarray()
    board = chess.Board(starting_fen)
    for move in moves:
        if debug:
            print('\t', end='')

        encoded_moves += encode_move(move,
                                     board,
                                     mask_legal=mask_legal,
                                     mask_pseudo_legal=mask_pseudo_legal,
                                     debug=debug)

        if debug:
            print('\n', end='')

        board.push(move)
    return encoded_moves


def decode_moves(bits: bitarray.bitarray,
                 starting_fen: str,
                 game_plies: int,
                 *,
                 mask_legal: bool = False,
                 mask_pseudo_legal: bool = False,
                 debug: bool = False,
                 debug_moves: List[chess.Move] = None) -> Tuple[List[chess.Move], int]:
    """ Return a list of moves decoded from ``bits``. See ``decode_move()``. """

    bits_consumed = 0

    board = chess.Board(starting_fen)
    for _ in range(game_plies):
        if debug:
            print('\t', end='')

        move, num_consumed = decode_move(bits,
                                         board,
                                         mask_legal=mask_legal,
                                         mask_pseudo_legal=mask_pseudo_legal,
                                         debug=debug)
        bits = bits[num_consumed:]
        bits_consumed += num_consumed

        if debug_moves is not None:
            if debug_moves[board.ply()] != move:
                move_num = format_fullmove(board.fullmove_number, board.turn)
                print(f' ERROR: decoded to {move_num}{move}, '
                      f'expected {move_num}{debug_moves[board.ply()]}')

        if debug:
            print('\n', end='')

        board.push(move)

    return board.move_stack, bits_consumed


if __name__ == '__main__':
    pgn = '''[Event "FIDE World Cup 2023"]
[Site "Baku AZE"]
[Date "2023.07.30"]
[Round "1.1"]
[White "Cheparinov, Ivan"]
[Black "Alhassadi, Yousef A."]
[Result "1-0"]
[WhiteTitle "GM"]
[WhiteElo "2663"]
[BlackElo "2030"]
[ECO "D30"]
[Opening "Queen's gambit declined"]
[WhiteFideId "2905540"]
[BlackFideId "9204725"]
[EventDate "2023.07.30"]
[EventType "k.o."]

1. d4 d5 2. c4 e6 3. Nf3 c5 4. cxd5 exd5 5. Bg5 Nf6 6. Nc3 Be6 7. a3 Nc6 8. e3
Be7 9. dxc5 Bxc5 10. Bb5 O-O 11. O-O Be7 12. Rc1 Rc8 13. Bh4 a6 14. Bxc6 bxc6
15. Qd3 Qb6 16. Rc2 Rfd8 17. Rfc1 Qa5 18. Nd4 c5 19. Nxe6 fxe6 20. e4 d4 21. Ne2
Qb5 22. Rc4 Qb7 23. h3 Kf7 24. e5 Qe4 25. Qxe4 Nxe4 26. Bxe7 Kxe7 27. f3 Nd2 28.
Rxc5 Rxc5 29. Rxc5 Rd7 30. Kf2 Nb3 31. Rc4 d3 32. Nc3 d2 33. Nd1 Nc1 34. Ke3 Nd3
35. Rd4 Rxd4 36. Kxd4 Ne1 37. Ke3 Nxg2+ 38. Kxd2 Kd7 39. Nf2 Kc6 40. Nd3 a5 41.
b4 axb4 42. Nxb4+ Kb5 43. Nd3 g5 44. Ke2 Nh4 45. f4 h6 46. fxg5 hxg5 47. Ke3 Kc4
48. Ke4 Nf5 49. a4 Ng3+ 50. Kf3 Nf5 51. a5 Kxd3 52. a6 Nd4+ 53. Kg4 Nc6 54. Kxg5
Ke4 55. h4 Kxe5 56. h5 1-0'''
    game = chess.pgn.read_game(io.StringIO(pgn))

    ITERS = 1000
    DEBUG_ENCODE = True
    DEBUG_DECODE = True

    moves = list(game.mainline_moves())
    starting_fen = game.board().fen()

    results: Dict[Tuple[bool, bool], Tuple[float, int]] = {}

    for mask_legal, mask_pseudo_legal in ((False, False), (False, True), (True, True)):
        print(f'mask_legal: {mask_legal}, mask_pseudo_legal: {mask_pseudo_legal}')
        with ContextTimer(5) as t:
            for _ in range(ITERS):
                encoded = encode_moves(moves,
                                       starting_fen,
                                       mask_legal=mask_legal,
                                       mask_pseudo_legal=mask_pseudo_legal,
                                       debug=DEBUG_ENCODE)
                decoded, consumed = decode_moves(encoded,
                                                 starting_fen,
                                                 len(moves),
                                                 mask_legal=mask_legal,
                                                 mask_pseudo_legal=mask_pseudo_legal,
                                                 debug=DEBUG_DECODE,
                                                 debug_moves=moves)

        print(f'Encoded: {encoded}')
        print(f'Encoded length: {len(encoded)}')
        print(f'Decoded: {decoded}')
        print(f'Consumed: {consumed}')
        print(f'Correct decoding: {moves == decoded}')
        print()

        results[(mask_legal, mask_pseudo_legal)] = t.time, len(encoded)

    # Get the baseline and compare the other results to it
    baseline = results[(False, False)]
    baseline_time, baseline_bits_used = baseline
    for (mask_legal, mask_pseudo_legal), (time, bits_used) in results.items():
        print(f'mask_legal: {mask_legal}, mask_pseudo_legal: {mask_pseudo_legal}')
        print(f'Time: {time:.3f} seconds')
        print(f'Bits used: {bits_used}')
        print(f'Time ratio to baseline: {time / baseline_time:.3f}')
        print(f'Bits used ratio to baseline: {bits_used / baseline_bits_used:.3f}')
        print()
