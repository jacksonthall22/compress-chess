import os

from user_options import MoveEncodingOption
from utils import to_uint8_bitarray, to_int, encode_str, decode_str, _pack_bin

from typing import Dict, Tuple
import argparse
from bitarray import bitarray
import chess.pgn


DEBUG_ENCODING = False
DEBUG_DECODING = False

COMMON_HEADERS = {
    'Event': 0,
    'Site': 1,
    'Date': 2,
    'Round': 3,
    'UTCDate': 4,
    'UTCTime': 5,
    'EndTime': 6,  # Chess.com specific
    'White': 7,
    'WhiteElo': 8,
    'WhiteTitle': 9,
    'WhiteTeam': 10,
    'WhiteFideId': 11,
    'Black': 12,
    'BlackElo': 13,
    'BlackTitle': 14,
    'BlackTeam': 15,
    'BlackFideId': 16,
    'Result': 17,
    'ECO': 18,
    'PlyCount': 19,
    'TimeControl': 20,
    'Termination': 21,
    'Annotator': 22,
    'FEN': 23,
    'Variant': 24,
    'SetUp': 25,
    'EventDate': 26,
    'Opening': 27,
    'Variation': 28,
}
HEADER_LENGTH_BYTES = 1
"""
Number of bytes used to encode the length of a header's value. Ex: If 1, 
only the first 255 bytes of a header's encoded name/value are encoded.
"""


def _encode_string_length_and_value(s: str, length_bytes: int = 1) -> bitarray:
    # Encode the actual value
    encoded_str = encode_str(s)
    encoded_value = bitarray()
    encoded_value.frombytes(encoded_str)

    max_bytes = 2 ** (length_bytes * 8) - 1

    if len(encoded_value) / 8 > max_bytes:
        raise ValueError(f'(TODO: Remove) Encoded string has {len(encoded_value) / 8} bytes, longer than '
                         f'{max_bytes} bytes: {s}')

    encoded_value = encoded_value[:max_bytes * 8]

    # Get a byte saying the number of bytes that come next in the encoded value
    assert len(encoded_value) % 8 == 0, 'Encoded value not byte-aligned'
    num_bytes = to_uint8_bitarray(len(encoded_value) // 8)

    if DEBUG_ENCODING:
        print(f'{num_bytes.to01()} ({int(num_bytes.to01(), 2)} bytes) '
              f'{encoded_value.to01()} ({encoded_str}) ', end='')

    return num_bytes + encoded_value


def _decode_header_length_and_value(bits: bitarray) -> Tuple[str, int]:
    """
    Decode the header name and value from the given ``bits``. Return
    the header value and the number of bits consumed to decode it.
    """
    # Get the first byte, which represents the number of bytes in the encoded value.
    # Then decode the value itself.
    length_bits = bits[:8]
    num_bytes = to_int(length_bits)
    value_bits = bits[8: 8 + num_bytes * 8]

    value = decode_str(value_bits.tobytes())
    consumed = 8 + num_bytes * 8

    if DEBUG_DECODING:
        print(f'{length_bits.to01()} ({num_bytes} bytes) '
              f'{value_bits.to01()} ({value})', end='')
    return value, consumed


def encode_headers(game: chess.pgn.Game) -> bitarray:
    encoded_headers = bitarray()

    '''
    The first several bits are reserved for common headers. This way we don't have to encode their names.
    
    If the common header is present, we encode it as follows:
        - A 1 bit, indicating its presence
        - A byte representing the length of the header's encoded value in bytes
        - The specified number of bytes encoding its value
    
    If the common header is not present, we encode it as follows:
        - A 0 bit, indicating its absence
    '''
    for header, i in COMMON_HEADERS.items():
        # A couple of special cases need to be handled manually
        if header == 'PlyCount' and 'PlyCount' not in game.headers:
            # Must be able to decode the correct number of plies later
            game.headers['PlyCount'] = str(game.end().ply())
        elif header == 'FEN' and 'FEN' not in game.headers:
            # Must be able to decode from the right starting position later.
            # If it's just the starting FEN, we can assume what it is while decoding.
            starting_fen = game.board().fen()
            if starting_fen != chess.STARTING_FEN:
                game.headers['FEN'] = starting_fen

        # Append a bit indicating whether the header is present or absent
        if header not in game.headers:
            encoded_headers.append(0)

            if DEBUG_ENCODING:
                print(f'0 ("{header}" absent)\n', end='')

            continue
        encoded_headers.append(1)

        if DEBUG_ENCODING:
            print(f'1 ("{header}" present) ', end='')

        # Encode the header value
        encoded_headers.extend(_encode_string_length_and_value(game.headers[header],
                                                               length_bytes=HEADER_LENGTH_BYTES))

        if DEBUG_ENCODING:
            print('\n', end='')

    '''
    Subsequent headers need to have their names encoded. Ex: for ``[SomeRandomTag "its value"]``, we need to
    encode the string "SomeRandomTag" and the string "its value".
    
    For every additional header, we encode it as follows: 
        - A 1 bit, indicating its presence
        - A byte representing the length of the header's encoded name
        - The specified number of bytes encoding its name (ex. "SomeRandomTag")
        - A byte representing the length of the header's encoded value
        - The specified number of bytes encoding its value (ex. "its value")
    '''
    # Skip empty headers
    for header in game.headers:
        if header in COMMON_HEADERS or not header:
            continue

        encoded_headers.append(1)

        if DEBUG_ENCODING:
            print(f'1 (next present) ', end='')

        encoded_headers.extend(_encode_string_length_and_value(header,
                                                               length_bytes=HEADER_LENGTH_BYTES))

        encoded_headers.extend(_encode_string_length_and_value(game.headers[header],
                                                               length_bytes=HEADER_LENGTH_BYTES))

        if DEBUG_ENCODING:
            print('\n', end='')

    '''
    After all headers have been encoded, indicate that we are done with a 0 bit.
    '''
    encoded_headers.append(0)

    if DEBUG_ENCODING:
        print(f'0 (no more headers)\n', end='')

    return encoded_headers


def decode_headers(bits: bitarray) -> Tuple[chess.pgn.Headers, int]:

    total_consumed = 0

    headers = {}
    for header, i in COMMON_HEADERS.items():
        if bits[0]:
            if DEBUG_DECODING:
                print(f'1 ("{header}" present) ', end='')

            bits = bits[1:]
            total_consumed += 1

            # Header value (already know the header name)
            value, consumed = _decode_header_length_and_value(bits)
            bits = bits[consumed:]
            total_consumed += consumed

            headers[header] = value
        else:
            if DEBUG_DECODING:
                print(f'0 ("{header}" absent)', end='')
            bits = bits[1:]
            total_consumed += 1

        if DEBUG_DECODING:
            print('\n', end='')

    # Custom / not-common headers
    while bits[0]:
        if DEBUG_DECODING:
            print(f'1 (next present) ', end='')

        bits = bits[1:]
        total_consumed += 1

        # Header name
        header, consumed = _decode_header_length_and_value(bits)
        if DEBUG_DECODING:
            print(f'{bits[:consumed].to01()} ({header}) ', end='')
        bits = bits[consumed:]
        total_consumed += consumed

        # Header value
        value, consumed = _decode_header_length_and_value(bits)
        if DEBUG_DECODING:
            print(f'{bits[:consumed].to01()} ({value})\n', end='')
        bits = bits[consumed:]
        total_consumed += consumed

        headers[header] = value

    # The final 0 bit
    total_consumed += 1

    if DEBUG_DECODING:
        print(f'0 (no more headers)\n', end='')

    return chess.pgn.Headers(headers), total_consumed


def encode_game(game: chess.pgn.Game,
                options: Dict[str, str | bool]) -> bitarray:
    """ Encode moves of a chess game to binary """

    # Start building the bitstring for the game
    bits = bitarray()

    ''' Encode the headers '''
    if DEBUG_ENCODING:
        print('Headers:\n', end='')

    bits.extend(encode_headers(game))


    ''' Encode the moves '''
    if DEBUG_ENCODING:
        print('Moves:\n', end='')

    # Pick a default move_encoding option if none is specified
    if 'move_encoding' not in options:
        options['move_encoding'] = MoveEncodingOption.handle_from_to_squares_separately

    # Handle the various cases
    if options['move_encoding'] == MoveEncodingOption.handle_from_to_squares_separately:
        from encode_from_to_squares import encode_moves
        mask_legal = options['mask_legal']
        mask_pseudo_legal = options['mask_pseudo_legal']
        bits.extend(encode_moves(game.mainline_moves(),
                                 game.board().fen(),
                                 mask_legal=mask_legal,
                                 mask_pseudo_legal=mask_pseudo_legal,
                                 debug=DEBUG_ENCODING))

    elif options['move_encoding'] == MoveEncodingOption.from_uci:
        for move in game.mainline_moves():
            bits.extend(_pack_bin(move.from_square, max_states=64))
            bits.extend(_pack_bin(move.to_square, max_states=64))

    else:
        raise NotImplementedError(f'move_encoding option "{options["move_encoding"]}" not implemented')

    # elif options['move_encoding'] == MoveEncodingOption.huffman_code_san:
    #     san_moves = [node.san() for node in game.mainline()]
    #     bits.extend(bitarray_util.huffman_code(san_moves))
    #     ...  # TODO
    #
    # elif options['move_encoding'] == MoveEncodingOption.map_to_action_space:
    #     ...  # TODO

    if DEBUG_ENCODING:
        print('\n', end='')

    return bits


def decode_game(bits: bitarray,
                options: Dict[str, str | bool]) -> Tuple[chess.pgn.Game, int]:

    total_consumed = 0

    # Decode the headers
    headers, consumed = decode_headers(bits)
    bits = bits[consumed:]
    total_consumed += consumed

    # Decode the moves
    if options['move_encoding'] == MoveEncodingOption.handle_from_to_squares_separately:
        from encode_from_to_squares import decode_moves
        mask_legal = options['mask_legal']
        mask_pseudo_legal = options['mask_pseudo_legal']
        try:
            game_plies = int(headers['PlyCount'])
        except KeyError:
            raise Exception('Internal error: "PlyCount" header should have been added while encoding if not present')

        # Assume the starting FEN if no FEN header provided
        starting_fen = headers.get('FEN', chess.STARTING_FEN)

        moves, consumed = decode_moves(bits,
                                       starting_fen,
                                       game_plies,
                                       mask_legal=mask_legal,
                                       mask_pseudo_legal=mask_pseudo_legal,
                                       debug=DEBUG_DECODING)
        total_consumed += consumed

        board = chess.Board(starting_fen)
        for move in moves:
            board.push(move)
        game = chess.pgn.Game.from_board(board)
        game.headers = headers
    else:
        raise NotImplementedError(f'move_encoding option "{options["move_encoding"]}" not implemented')

    return game, total_consumed


def encode_pgn_file(file: str,
                    output_file: str,
                    options: Dict[str, str | bool]) -> bitarray:
    """ Encode a PGN file to binary """

    bits = bitarray()

    # Load and encode all games
    with open(file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            bits.extend(encode_game(game, options))

    if output_file is not None:
        with open(output_file, 'wb') as f:
            bits.tofile(f)

    if DEBUG_ENCODING:
        print(f'Encoded "{file}" to "{output_file}" '
              f'({os.path.getsize(file) / os.path.getsize(output_file):.2f}x compression)')

    return bits


def decode_pgn_file(file: str,
                    output_file: str,
                    options: Dict[str, str | bool]) -> Tuple[str, int]:
    """ Decode a binary file to a PGN. Return the PGN string and the number of bits consumed while decoding. """

    total_consumed = 0

    # Load and decode all games
    with open(file, 'rb') as f:
        bits = bitarray()
        bits.fromfile(f)

    games = []
    while bits:
        game, consumed = decode_game(bits, options)
        games.append(game)

        bits = bits[consumed:]
        total_consumed += consumed

    pgn = '\n\n'.join(map(str, games))

    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(pgn)

    if DEBUG_ENCODING:
        print(f'Decoded "{file}" to "{output_file}"')

    return pgn, total_consumed


def encode_pgn_dir(dir: str,
                   output_dir: str,
                   options: Dict[str, str | bool]) -> str:
    """ Encode all PGN files in a directory to binary files in a new directory. Returns ``output_dir``. """

    files = os.listdir(dir)
    for file in files:
        if file.lower().endswith('.pgn'):
            output_file = os.path.join(output_dir, file + '.bin')
            encode_pgn_file(file, output_file, options)

    return output_dir


def decode_pgn_dir(dir: str,
                   output_dir: str,
                   options: Dict[str, str | bool]) -> str:
    """ Decode all binary files in a directory to PGNs in a new directory. Returns ``output_dir``. """

    files = os.listdir(dir)
    for file in files:
        if file.lower().endswith('.bin'):
            output_file = os.path.join(output_dir, file[:-4] + '.pgn')
            decode_pgn_file(file, output_file, options)

            print(f'Decoded {file} to {output_file} '
                  f'({os.path.getsize(file) / os.path.getsize(output_file):.2f}x compression)')

    return output_dir


# def main(args: argparse.Namespace):
#     if args.file is not None:
#         encode_pgn_file(args.file, args.output_path)



def main():
    parser = argparse.ArgumentParser(prog='bitpgn',
                                     description='Compress PGN files with various strategies.')

    ''' Specifying whether to encode or decode '''
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--encode',
                              '-e',
                              action='store_true',
                              help='Encode a PGN file to binary')
    action_group.add_argument('--decode',
                              '-d',
                              action='store_true',
                              help='Decode a binary file to a PGN')

    ''' Providing the input file or directory '''
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file',
                             '-f',
                             help='Input file to compress/decompress')
    input_group.add_argument('--path',
                             '-p',
                             help='Input directory containing PGN/binary files to compress/decompress')

    ''' Providing the output file or directory '''
    parser.add_argument('--output-file',
                        '-F',
                        help='Output file for the compressed/decompressed binary/PGN file. '
                             'File will be overwritten.')
    parser.add_argument('--output-path',
                        '-P',
                        help='Output directory for the compressed/decompressed binary/PGN files. '
                             'Directory will be wiped and overwritten.')

    ''' Encoding options '''
    parser.add_argument('--move-encoding', '-m',
                        type=MoveEncodingOption,
                        default=MoveEncodingOption.handle_from_to_squares_separately,
                        help='Strategy for encoding/decoding moves to binary')

    masking_group = parser.add_mutually_exclusive_group()
    masking_group.add_argument('--mask-legal',
                               dest='mask_legal',
                               action='store_true',
                               help='Enable masking legal moves. '
                                    'Only for the "handle_from_to_squares_separately" strategy.')
    masking_group.add_argument('--mask-pseudo-legal',
                               dest='mask_pseudo_legal',
                               action='store_true',
                               help='Enable masking pseudo-legal moves. '
                                    'Only for the "handle_from_to_squares_separately" strategy.')

    parser.add_argument('--debug',
                        '-D',
                        action='store_true',
                        help='Enable debug mode',
                        default=False)

    args = parser.parse_args()

    ''' Input validation '''
    if args.file and args.path:
        parser.error('You cannot provide both input file and input directory.')

    if args.output_file and not args.file:
        parser.error('You can only specify an output file if you provide an input file.')

    if args.output_path and not args.path:
        parser.error('You can only specify an output directory if you provide an input directory.')

    # If no output file or directory is provided, generate filenames here.
    if not args.output_file:
        if args.encode:
            postfix = '_encoded.bin'
        else:
            assert args.decode
            postfix = '_decoded.pgn'
        args.output_file = os.path.splitext(args.file)[0] + postfix if args.file else None

    if not args.output_path:
        if args.encode:
            postfix = '_encoded'
        else:
            assert args.decode
            postfix = '_decoded'
        args.output_path = os.path.dirname(args.path) + postfix if args.path else None

    if args.move_encoding != MoveEncodingOption.handle_from_to_squares_separately:
        if args.mask_legal is not None:
            parser.error('Masking options can only be set for the `handle_from_to_squares_separately` strategy.')

    # Your compression logic goes here.
    # args.file will contain the input PGN file (if provided).
    # args.path will contain the input directory (if provided).
    # args.output_file will contain the output compressed PGN file (if provided).
    # args.output_path will contain the output directory (if provided).

    global DEBUG_ENCODING
    DEBUG_ENCODING = args.debug

    if args.encode:
        if args.file is not None:
            encode_pgn_file(args.file, args.output_file, vars(args))
        elif args.path is not None:
            encode_pgn_dir(args.path, args.output_path, vars(args))
    else:
        assert args.decode
        if args.file is not None:
            decode_pgn_file(args.file, args.output_file, vars(args))
        elif args.path is not None:
            decode_pgn_dir(args.path, args.output_path, vars(args))


if __name__ == "__main__":
    main()

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog='bitpgn',
#         description='Encode and decode a series of chess moves from a PGN into an '
#                     'efficient binary representation for long-term storage.')
#     subparsers = parser.add_subparsers(help='sub-command help')
#
#     ''' Encode a PGN file '''
#     parser_encode = subparsers.add_parser('encode', help='Encode a PGN file')
#
#     # Specify options for move encoding
#     parser_encode.add_argument('-m',
#                                '--move-encoding',
#                                type=MoveEncodingOption,
#                                default=MoveEncodingOption.handle_from_to_squares_separately,
#                                help='Encoding method for moves')
#
#
#     # User should only be allowed to encode a single file or a directory of files, not both.
#     # Code below adds a mutually exclusive group
#     group = parser_encode.add_mutually_exclusive_group()
#     group.add_argument('-f', '--file', type=int, help='A single PGN file to encode')
#     group.add_argument('-d', '--dir', type=int, help='A directory of PGN files to encode')
#
#     group.add
#
#
#     parser = argparse.ArgumentParser(description='Encode and decode a series of chess moves from a PGN into an '
#                                                  'efficient binary representation for long-term storage.')
#     parser.add_argument('-f', '--file', type=str, help='A single PGN file to encode')
#     parser.add_argument('-i', '--input-dir', type=int, help='A directory of PGNs to encode')
#     parser.add_argument('-o', '--output-dir', type=int, help='A directory to store the encoded PGNs')
#     parser.add_argument('-m', '--max-plies', type=int, help='The maximum number of plies to encode')
#     parser.add_argument('-p', '--parallel', type=int, help='The number of parallel processes to use')
#     parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
#
#     args = parser.parse_args()
#
#     # Convert args to a dict
#     options = vars(args)
#
#     # Remove None values
#     options = {k: v for k, v in options.items() if v is not None}
#
#     # Remove the verbose flag
#     verbose = options.pop('verbose')
#
#
#     main(**options)
#
#
#
