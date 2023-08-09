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


def encode_move(move: chess.Move,
                board: chess.Board,
                *,
                mask_legal: bool = False,
                mask_pseudo_legal: bool = False) -> bitarray.bitarray:
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
    return encoded_from_sq + encoded_to_sq


def decode_move(bits: bitarray.bitarray,
                board: chess.Board,
                *,
                mask_legal: bool = False,
                mask_pseudo_legal: bool = False) -> Tuple[chess.Move, int]:
    """
    Decode the first bits in ``bits`` into a move by decoding ``from_square`` and ``to_square`` based on
    ``board``. Return the decoded move and the number of bits consumed to decode it. See ``decode_from_square()``
    and ``decode_to_square()``.
    """
    movable_pieces_bb = _get_movable_pieces_bb(board,
                                               mask_legal=mask_legal,
                                               mask_pseudo_legal=mask_pseudo_legal)
    from_square_bit_len = bits_required_for_n_states(movable_pieces_bb.bit_count())

    from_square = decode_from_square(bits[:from_square_bit_len],
                                     board,
                                     mask_legal=mask_legal,
                                     mask_pseudo_legal=mask_pseudo_legal)

    actions_bb = _get_actions_bb(from_square,
                                 board,
                                 mask_legal=mask_legal,
                                 mask_pseudo_legal=mask_pseudo_legal)

    to_square_bit_len = bits_required_for_n_states(actions_bb.bit_count())
    to_square = decode_to_square(from_square,
                                 bits[from_square_bit_len:from_square_bit_len + to_square_bit_len],
                                 board,
                                 mask_legal=mask_legal,
                                 mask_pseudo_legal=mask_pseudo_legal)

    return chess.Move(from_square, to_square), from_square_bit_len + to_square_bit_len


def encode_moves(moves: Iterable[chess.Move],
                 starting_fen: str,
                 *,
                 mask_legal: bool = False,
                 mask_pseudo_legal: bool = False) -> bitarray.bitarray:
    """ Return a concatenation of each encoded move in ``moves``. See ``encode_move()``. """
    encoded_moves = bitarray.bitarray()
    board = chess.Board(starting_fen)
    for move in moves:
        encoded_moves += encode_move(move,
                                     board,
                                     mask_legal=mask_legal,
                                     mask_pseudo_legal=mask_pseudo_legal)
        board.push(move)
    return encoded_moves


def decode_moves(bits: bitarray.bitarray,
                 starting_fen: str,
                 *,
                 mask_legal: bool = False,
                 mask_pseudo_legal: bool = False,
                 debug_moves: List[chess.Move] = None) -> Tuple[List[chess.Move], int]:
    """ Return a list of moves decoded from ``bits``. See ``decode_move()``. """

    bits_consumed = 0

    board = chess.Board(starting_fen)
    while bits:
        move, num_consumed = decode_move(bits,
                                         board,
                                         mask_legal=mask_legal,
                                         mask_pseudo_legal=mask_pseudo_legal)
        bits = bits[num_consumed:]
        bits_consumed += num_consumed

        if debug_moves is not None:
            if debug_moves[board.ply()] != move:
                move_num = format_fullmove(board.fullmove_number, board.turn)
                print(f'Error: decoded to {move_num}{move}, '
                      f'expected {move_num}{debug_moves[board.ply()]}')

        board.push(move)

    return board.move_stack, bits_consumed


if __name__ == '__main__':
    pgn = '''[Event "Casual Bullet game"]
[Site "https://lichess.org/Y2hanlPl"]
[Date "2022.07.01"]
[White "Goldy35"]
[Black "penguingm1"]
[Result "0-1"]
[UTCDate "2022.07.01"]
[UTCTime "21:10:02"]
[WhiteElo "1133"]
[BlackElo "2481"]
[BlackTitle "GM"]
[Variant "Standard"]
[TimeControl "60+0"]
[ECO "D31"]
[Opening "Queen's Gambit Declined: Charousek Variation"]
[Termination "Time forfeit"]
[Annotator "lichess.org"]

1. d4 d5 2. c4 e6 3. Nc3 Be7 { D31 Queen's Gambit Declined: Charousek Variation } 4. Nf3 Nf6 5. cxd5 exd5 6. Bf4 c6 7. e3 Bf5 8. Be2 Nbd7 9. O-O O-O 10. h3 h6 11. Rc1 Re8 12. Re1 Nf8 13. e4?? { (0.13 → -2.18) Blunder. Ne5 was best. } (13. Ne5 Bd6 14. Bd3 Bxd3 15. Nxd3 Bxf4 16. Nxf4 Ne6 17. Nd3 a5 18. a3 Re7 19. b4 Nc7) 13... Nxe4 14. Bd3 Nxc3 15. bxc3 Bxd3 16. Qxd3 Ne6?! { (-1.70 → -0.85) Inaccuracy. Bf6 was best. } (16... Bf6 17. Be5) 17. Bh2?! { (-0.85 → -1.93) Inaccuracy. Rxe6 was best. } (17. Rxe6 fxe6) 17... Bd6 18. Bxd6 Qxd6 19. Ne5 Re7 20. Re3 Rae8 21. Rce1 Nf4?! { (-2.06 → -1.07) Inaccuracy. Ng5 was best. } (21... Ng5) 22. Qf5 Ne6 23. f4?! { (-1.43 → -2.50) Inaccuracy. h4 was best. } (23. h4 Nf8) 23... Nf8 24. Qd3 f6 25. Nxc6?? { (-2.82 → -7.78) Blunder. Nf3 was best. } (25. Nf3) 25... bxc6 26. Rxe7 Rxe7 27. Rxe7 Qxe7 { Black wins on time. } 0-1'''
    game = chess.pgn.read_game(io.StringIO(pgn))

    moves = list(game.mainline_moves())
    starting_fen = game.board().fen()

    results: Dict[Tuple[bool, bool], Tuple[float, int]] = {}

    for mask_legal, mask_pseudo_legal in ((False, False), (False, True), (True, True)):
        print(f'mask_legal: {mask_legal}, mask_pseudo_legal: {mask_pseudo_legal}')
        with ContextTimer(5) as t:
            for _ in range(1000):
                encoded = encode_moves(moves,
                                       starting_fen,
                                       mask_legal=mask_legal,
                                       mask_pseudo_legal=mask_pseudo_legal)
                decoded, consumed = decode_moves(encoded,
                                                 starting_fen,
                                                 mask_legal=mask_legal,
                                                 mask_pseudo_legal=mask_pseudo_legal,
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
