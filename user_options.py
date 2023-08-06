from enum import StrEnum


class MoveEncodingOption(StrEnum):
    from_uci = 'from_uci'
    """
    Encode using the naive approach, using 12 bits per move: 6 for the move's
    ``from_square`` and 6 for its ``to_square``.
    """

    huffman_code_uci = 'huffman_code_uci'
    """
    Encode moves with a Huffman code based on the UCI string of all moves in the game.
    """

    huffman_code_san = 'huffman_code_san'
    """
    Encode moves with a Huffman code based on the SAN string of all moves in the game.
    """

    map_to_action_space = 'map_to_action_space'
    """
    Map the move to an 11-bit representation based on chess's 
    `discrete action space <https://www.google.com>`_ of size 1924.
    """

    map_to_action_space_2fold_symmetry = 'map_to_action_space_2fold_symmetry'
    """
    Map moves that are symmetric about 4th/5th rank line to a 10-bit 
    representation, assuming the corresponding symmetric move is impossible, 
    based on chess's `discrete action space <https://www.google.com>`_. 
    Fall back to the 11-bit representation if both moves are possible.
    """

    map_to_action_space_4fold_symmetry = 'map_to_action_space_4fold_symmetry'
    """
    Map moves that are symmetric about the 4th/5th rank line or the d/e file line
    to a 9-bit representation, assuming all corresponding symmetric moves are impossible,
    based on chess's `discrete action space <https://www.google.com>`_.
    Fall back to the 10-bit representation, or the 9-bit representation if another 
    symmetric move is possible.
    """

    '''
    TODO: After lots of unexpected roadblocks, I'm abandoning this idea for now. The number of 8-bit representations 
          necessary turns out to be 260, just over 256. It's because moves on the long diagonal must be assigned an
          8-bit binary prefix, but then do not make full use of the subsequent 3 bits, since they only have 4 moves
          in their symmetry group instead of 8. This means we run out of 8-bit prefixes too early, counterintuitively.
          Here's the math:
          
            - There are 1924 actions total (see my chess-action-space repo)
            - Moves on either long diagonal have only 4 moves in their symmetry group - there are 
              8 * 7 * 2 = 112 of those. We need 112 / 4 = 28 8-bit base numbers to represent them.
            - The underpromotion action space is (8 + 7 + 7) moves * 2 sides * 3 pieces = 132, but by checking the
              turn while encoding/decoding, we can effectively reduce it by half to 66. It would be nice to assign
              each underpromotion's (from_sq, to_sq) pair an 8-bit base so we can easily distinguish them from 
              non-underpromotions, including regular moves and the "default" queen promotion, since then we
              could use the next 2 bits to specify the underpromotion piece. This means we should conceptualize
              underpromotion moves as having no symmetries: each movement is unique. So we need to consume 
              (8 + 7 + 7) = 22 more 8-bit bases here. The total is now 28 + 22 = 50.
            - All other moves of the 1924 have 8 moves in their symmetry group. There are 1924 - 112 - 132 = 1680
              of them, so we need 1680 / 8 = 210 8-bit bases to represent them. The final total is 260, just over 
              2 ** 8 = 256.
              
          To fix, we would have to be smart about which moves consume 9 bits. While encoding and decoding, we would
          have to know based on only the first 8 bits whether that base is one that requires 9 bits to distinguish.
          We cannot just continue assigning the binary representation of a gensym-ed number once it hits 9 bits long
          because the first 8 of those bits will already have been generated and assigned to some other move.
          
          We should probably choose 4 relatively uncommon base moves (from 4 different symmetry groups) whose 8-bit 
          prefixes make up a special case, where a 9th bit is read to distinguish the base move. Then the same scheme 
          applies where the next 3 bits determine the symmetries to be applied to the base move.
          
          Another option, which seems inherently worse, is to use another bit to represent moves where the 
          (from_sq, to_sq) pair is the same as that of an underpromotion move for the side to move. If the bit is 
          a 0, we can say the move is a non-underpromotion; if it's a 1, it's an underpromotion and the next 2 bits 
          should determine the piece. I don't think this will offer maximal compression since it requires an extra bit
          each time a piece is moved from the 7th to the 8th rank (except for knights) from the POV of the side to move
          (not that uncommon, especially in endgames).
          
          A hacky option is to not care about ex. capturing underpromotions onto a corner squares, which would 
          free 4 of the 260 8-bit bases. This means [b7a8, b2a1, g7h8, g2h1]=[R, B, N] would encode to 
          [b7a8, b2a1, g7h8, g2h1]=Q. We surely wouldn't miss many moves, but the ones we do might be brilliant.
    '''
    # map_to_action_space_8fold_symmetry = 'map_to_action_space_8fold_symmetry'
    # """
    # Map moves that are symmetric about the 4th/5th rank line, the d/e file line,
    # and the a1-h8 diagonal to an 8-bit representation, assuming all corresponding
    # symmetric moves are impossible, based on chess's `discrete action space
    # <https://www.google.com>`_. Fall back to the 9-bit, 10-bit, or 11-bit representation
    # if another symmetric move is possible.
    # """

    handle_from_to_squares_separately = 'handle_from_to_squares_separately'
    """
    Provide additional options to encode ``from_square`` and ``to_square`` separately.
    """


class FromSquareEncodingOption(StrEnum):
    occupied_index = 'occupied_index'
    """
    Encode ``move.from_square`` based on its index out of the 
    truthy bits in ``board.occupied_co[board.turn()]``.
    """

    mask_legal = 'mask_legal'
    """
    Encode ``move.from_square`` based on its index out of truthy bits in the 
    bitboard that unions all of ``chess.BB_SQUARES[m.from_square]``
    for legal moves ``m`` on the board.
    """

    mask_pseudo_legal = 'mask_pseudo_legal'
    """
    Encode ``move.from_square`` based on its index out of truthy bits in the
    bitboard that unions all of ``chess.BB_SQUARES[move.from_square]``
    for pseudo-legal moves ``m`` on the board. Significantly faster than
    the ``mask_legal`` option, with slightly worse bit-savings.
    """

    square_index = 'square_index'
    """
    Encode ``move.from_square`` as a ``0–63`` integer using 6 bits. 
    """


class ToSquareEncodingOption(StrEnum):
    mask_piece_square_action_space = 'mask_potential_legal'
    """
    Encode ``move.to_square`` using a pre-generated bitboard with truthy bits
    for all squares that are potential legal moves or captures for the piece at
    ``from_square``.
    """

    mask_legal = 'mask_legal'
    """
    Encode ``move.to_square`` based on its index out of truthy bits in the
    bitboard that unions all of ``chess.BB_SQUARES[move.to_square]``
    for legal moves ``m`` on the board.
    """

    mask_pseudo_legal = 'mask_pseudo_legal'
    """
    Encode ``move.to_square`` based on its index out of truthy bits in the
    bitboard that unions all of ``chess.BB_SQUARES[move.to_square]``
    for pseudo-legal moves ``m`` on the board. Significantly faster than
    the ``mask_legal`` option, with slightly worse bit-savings.
    """
