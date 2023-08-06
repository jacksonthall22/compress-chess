from bitarray import bitarray
import chess


STR_ENCODING = 'utf8'


def format_fullmove(fullmove: int, color: chess.Color, /) -> str:
    """ Return a string representation of the fullmove number and color. """
    return f'{fullmove}.{".." if color == chess.BLACK else ""}'


def _pack_bin(value: int, /, *, max_states: int) -> bitarray:
    """ Return a bitarray representation of ``value`` padded with the appropriate number of 0s given ``max_states``. """
    assert value < max_states, '``value`` must be less than ``max_states``'

    if value <= 1:
        return bitarray()
    return bitarray(format(value, f'0{(max_states-1).bit_length()}b'))


def to_int(bits: bitarray) -> int:
    """ Efficiently convert a bitarray to an int. """
    assert bits.endian() == 'big'
    i = 0
    for bit in bits:
        i = (i << 1) | bit
    return i


def bits_required_for_n_states(states: int, /) -> int:
    """
    Return the number of bits needed to represent ``states`` unique states. Slightly different
    from ``int.bit_length(states)``, since 000... is a valid state.
    Note: Returns 0 if ``states == 0`` or ``states == 1``.
    """
    return (states - 1).bit_length()


# def concatenate_bits(a: int,
#                      b: int,
#                      *,
#                      min_len_a: int = 0,
#                      min_len_b: int = 0) -> int:
#     # TODO not working (ChatGPT)
#     num_bits_a = max(a.bit_length(), min_len_a)
#     num_bits_b = max(b.bit_length(), min_len_b)
#
#     shift_amount = num_bits_b
#
#     a_shifted = a << shift_amount
#     result = a_shifted | b
#     return result
def encode_str(s: str) -> bytes:
    return s.encode(STR_ENCODING)


def decode_str(encoded_s: bytes) -> str:
    return encoded_s.decode(STR_ENCODING)
