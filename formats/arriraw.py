from sys import argv
from enum import Enum
from pathlib import Path
from collections import namedtuple
from struct import unpack_from

from argparse import ArgumentParser

import numpy as np

from fileseq import findSequencesOnDisk

import OpenImageIO as oiio
from OpenImageIO import ImageSpec, ImageOutput
# from OpenImageIO import ImageInput  # debugging


class Pattern(Enum):
    GRBG = 0
    GBRG = 1
    BGGR = 2
    RGGB = 3


class Site(Enum):
    G1 = 1
    R = 2
    B = 3
    G2 = 4
    
    
ROOT_STRUCT_OFFSET = 0x0
ROOT_STRUCT_FORMAT = '<IIII'
ROOT_NAME = 'RootInfo'
ROOT_MEMBERS = ['MagicNum', 'ByteOrder', 'HeaderSize', 'VersionNumber']

IDI_STRUCT_OFFSET = 0x10
IDI_STRUCT_FORMAT = '<IIIIIIIIIIIIIIIHH'
IDI_NAME = 'ImageDataInfo'
IDI_MEMBERS = ['Valid', 'Width', 'Height', 'DataType', 'DataSpace',
               'ActiveImageLeft', 'ActiveImageTop', 'ActiveImageWidth', 'ActiveImageHeight',
               'FullImageLeft', 'FullImageTop', 'FullImageWidth', 'FullImageHeight',
               'ImageDataOffset', 'ImageDataSize',
               'SensorReadoutOffsetHorizontal', 'SensorReadoutOffsetVertical']

ICI_STRUCT_OFFSET = 0x54
ICI_STRUCT_FORMAT = (
        '<III'          # Valid, Version, WhiteBalance
        + 'ffff'        # GreenTintFactor, WhiteBalanceFactorR,WhiteBalanceFactorG, WhiteBalanceFactorB
        + 'IIII'        # WBAppliedInCameraFlag, ExposureIndex, BlackLevel, WhiteLevel
        + '12f'         # ColorMatrix,
        + 'ff'          # ColorMatrixDesatGain, ColorMatrixDesatOffset
        + 'III'         # HighlightDesaturationFlag, TargetColorSpace, Sharpness
        + 'f'           # PixelAspectRatio
        + 'I'           # Flip
        + '32s'         # LookFile
        + 'IIII'        # LookLutMode, LookLutOffset, LookLutSize, LookLiveGradingFlags
        + 'f'           # Saturation
        + '3f3f3f3f'    # CdlSlope, CdlOffset, CdlPower, PrinterLights
        + 'III'         # CdlApplicationMode, ImageDataChecksumType, ImageDataChecksum
        + '4s'          # ColorOrder
        + 'I'           # ColorimetricDataSetType
        + 'II'          # CameraSpecificColorimetricDataOffset, ColorimetricDataSetTypeSize
        + 'I'           # ColorimetricDataSetTypeCRC
)
ICI_NAME = 'ImageContentInformation'
ICI_MEMBERS = [
    'Valid', 'Version', 'WhiteBalance',
    'GreenTintFactor', 'WhiteBalanceFactorR', 'WhiteBalanceFactorG', 'WhiteBalanceFactorB',
    'WBAppliedInCameraFlag', 'ExposureIndex', 'BlackLevel', 'WhiteLevel',
    'a00', 'a01', 'a02', 'a04', 'a10', 'a11', 'a12', 'a13', 'a20', 'a21', 'a22', 'a23',
    'ColorMatrixDesatGain', 'ColorMatrixDesatOffset',
    'HighlightDesaturationFlag',  'TargetColorSpace', 'Sharpness',
    'PixelAspectRatio',
    'Flip',
    'LookFile',
    'LookLutMode', 'LookLutOffset', 'LookLutSize', 'LookLiveGradingFlags',
    'Saturation',
    'cdl_sr', 'cdl_sg', 'cdl_sb', 'cdl_or', 'cdl_og', 'cdl_ob', 'cdl_pr', 'cdl_pg', 'cdl_pb',
    'pl_r', 'pl_g', 'pl_b',
    'CdlApplicationMode', 'ImageDataChecksumType', 'ImageDataChecksum',
    'ColorOrder',
    'ColorimetricDataSetType',
    'CameraSpecificColorimetricDataOffset', 'CameraSpecificColorimetricSize',
    'CameraSpecificColorimetricDataCRC'
]

ALEXA_COLOR_ORDER = 'GRBG'
ALEXA65_COLOR_ORDER = 'GBRG'

CHUNK_SIZE_IN_BYTES = 12
PHOTOSITES_PER_CHUNK = 8


def image_desc_structs(ari):
    header = ari.read(4096)
    RootInfo = namedtuple(ROOT_NAME, ' '.join(ROOT_MEMBERS))
    ri = RootInfo(*unpack_from(ROOT_STRUCT_FORMAT, header, ROOT_STRUCT_OFFSET))
    ImageDataInfo = namedtuple(IDI_NAME, ' '.join(IDI_MEMBERS))
    idi = ImageDataInfo(*unpack_from(IDI_STRUCT_FORMAT, header, IDI_STRUCT_OFFSET))
    ImageContentInfo = namedtuple(ICI_NAME, ' '.join(ICI_MEMBERS))
    ici = ImageContentInfo(*unpack_from(ICI_STRUCT_FORMAT, header, ICI_STRUCT_OFFSET))
    return ri, idi, ici


def chunk_bounds_and_photosite_buf_start(stride, min_x, max_x, y):
    """
    Determine which chunks are required for a given run of photosites, and in the 1-D array
    of uint16 that results from the unpacking of those chunks, the index of the element
    that corresponds to min_x.

    A 'chunk' is a 12-byte sequence of packed data, containing eight photosites worth of
    data.

    n.b. min_x is inclusive, max_x is exclusive, so 43, 47 means [43, 47) means 43, 44, 45, 46.

    Parameters
    ----------
    stride : int
        width of photosite array, i.e. the amount of photosites per line
    min_x : int
        inclusive zero-based x-coordinate of photosite on line y
    max_x : int
        exclusive zero-based x-coordinate of photosite on line y
    y: int
        zero-based y-coordinate of the line the data of which we seek. Origin is at top left.

    Returns
    -------
    starting_chunk : int
        zero-based number of chunk containing min_x photosite data
    ending_chunk : int
        zero-based number of first chunk after starting_chunk that does NOT contain desired photosite data
    min_x_photosite_offset : int
        number of photosites into the decoded photosite array where one finds data for min_x photosite

    """
    min_x_photosite_offset_from_origin = y * stride + min_x
    max_x_photosite_offset_from_origin = y * stride + max_x
    start_chunk = min_x_photosite_offset_from_origin // PHOTOSITES_PER_CHUNK
    if max_x == min_x:
        end_chunk = start_chunk
    else:
        end_chunk = max_x_photosite_offset_from_origin // PHOTOSITES_PER_CHUNK
        if max_x_photosite_offset_from_origin % PHOTOSITES_PER_CHUNK != 0:
            end_chunk = end_chunk + 1
    # calculate the x position of the first photosite decoded from the first chunk
    leftmost_photosite_in_starting_chunk = (start_chunk * PHOTOSITES_PER_CHUNK) % stride
    x_photosite_offset = min_x - leftmost_photosite_in_starting_chunk
    assert x_photosite_offset >= 0
    return start_chunk, end_chunk, x_photosite_offset


def fill_photosites_from_chunks(chunk_buf, num_chunks, photosite_buf):
    for chunk_num in range(num_chunks):
        for photosite in range(PHOTOSITES_PER_CHUNK):
            for nib in range(3):
                nibs_from_chunk_start = photosite * 3 + nib
                src_ix = nibs_from_chunk_start // 8
                src_shift = 28 - 4 * (nibs_from_chunk_start % 8)
                dst_ix = chunk_num * PHOTOSITES_PER_CHUNK + photosite
                if dst_ix % 2 == 0:
                    dst_ix = dst_ix + 1
                else:
                    dst_ix = dst_ix - 1
                dst_shift = 4 * (2 - nib)
                photosite_buf[dst_ix] |= (((chunk_buf[src_ix] >> src_shift) & 0x0F) << dst_shift)


def fill_result_from_photosites(min_x, max_x, photosite_buf, photosite_buf_start,
                                result_buf, result_buf_offset, result_y):
    ps_start = photosite_buf_start
    ps_end = ps_start + max_x - min_x
    re_start = result_buf_offset
    re_end = re_start + max_x - min_x
    result_buf[result_y][re_start:re_end] = photosite_buf[ps_start:ps_end]
    print('should be filled-in now')


def photosites_from_file(ari_path, min_x, max_x, min_y, max_y, verbose=False):
    with open(ari_path, 'rb') as ari:
        ri, idi, ici = image_desc_structs(ari)
        if min_x < 0:
            print(f"minimum x value (inclusive) {min_x} is negative, setting to 0")
            min_x = 0
        if max_x >= idi.Width:
            print(f"maximum x value (exclusive) {max_x} exceeds image width {idi.Width}, setting to {idi.Width}")
            max_x = idi.Width
        if min_y < 0:
            print(f"minimum y value (inclusive) {min_y} is negative, setting to 0")
            min_y = 0
        if max_y >= idi.Height:
            print(f"maximum y value (exclusive) {max_y} exceeds image height {idi.Height}, setting to {idi.Height}")
            max_y = idi.Height
        if verbose:
            print(f"idi.Width is {idi.Width}")
            print(f"idi.Height is {idi.Height}")
            print(f"idi.ActiveImageWidth is {idi.ActiveImageWidth}")
            print(f"idi.ActiveImageHeight is {idi.ActiveImageHeight}")
            print('Extraction window:')
            print(f"  [{min_x}...{max_x}) -> {max_x - min_x} photosites wide")
            print(f"  [{min_y}...{max_y}) -> {max_y - min_y} photosites tall")
            print(f"Image data offset in .ari file: {idi.ImageDataOffset}")
        result_width = max_x - min_x  # [min_x, max_x)
        result_height = max_y - min_y  # [min_y, max_y)
        result_buf = np.zeros((result_height, result_width), dtype=np.uint16)
        file_cursor = ari.tell()
        for result_y, y in enumerate(range(min_y, max_y)):
            if verbose:
                print(f"{file_cursor=}")
            start_chunk, end_chunk, photosite_buf_start = chunk_bounds_and_photosite_buf_start(idi.Width, min_x, max_x, y)
            start_chunk_file_position = idi.ImageDataOffset + start_chunk * CHUNK_SIZE_IN_BYTES
            file_cursor_increment_needed = start_chunk_file_position - file_cursor
            print(f"{result_y=}, {start_chunk_file_position=}, {file_cursor_increment_needed=}")
            num_chunks = end_chunk - start_chunk  # because the interval is closed at bottom, open at the top
            photosite_buf = np.zeros(num_chunks * PHOTOSITES_PER_CHUNK, dtype=np.uint16)
            chunk_buf = np.fromfile(ari,
                                    dtype=np.dtype('<i4'),
                                    offset=file_cursor_increment_needed,
                                    count=num_chunks * 3)  # three uint32s per chunk
            file_cursor = start_chunk_file_position
            fill_photosites_from_chunks(chunk_buf, num_chunks, photosite_buf)
            fill_result_from_photosites(min_x, max_x, photosite_buf, photosite_buf_start, result_buf, 0, result_y)
        return result_buf


def inverse_qlut(buffer):
    ix = buffer >= 1024
    buffer[ix] = ((1024 + 2 * (buffer[ix] % 512) + 1) << ((buffer[ix] // 512) - 2)) - 1
    return buffer


def write_photosite_region_as_img(pb, min_x, min_y, color_order, img_path):
    output = ImageOutput.create(filename=str(img_path))
    if not output:
        raise RuntimeError(f"could not create output file `{img_path}'")
    height = pb.shape[0]
    width = pb.shape[1]
    pb = inverse_qlut(pb)
    spec = ImageSpec(width, height, 3, oiio.TypeFloat)
    if not output.open(str(img_path), spec):
        raise RuntimeError(f"could not open output file `{img_path}'")
    rgb_img = np.zeros((height, width, 3), dtype=float)
    origin = '??'
    if color_order == ALEXA_COLOR_ORDER:  # GRBG
        # what kind of photosite is at photosite region origin?
        if min_y % 2 == 0:  # even-numbered row
            if min_x % 2 == 0:
                origin = 'G1'
                g1_x_offset = 0
                g1_y_offset = 0
                r_x_offset = 1
                r_y_offset = 0
                b_x_offset = 0
                b_y_offset = 1
                g2_x_offset = 1
                g2_y_offset = 1
            else:
                origin = 'R'
                g1_x_offset = 1
                g1_y_offset = 0
                r_x_offset = 0
                r_y_offset = 0
                b_x_offset = 1
                b_y_offset = 1
                g2_x_offset = 0
                g2_y_offset = 1
        else:  # odd-numbered row
            if min_x % 2 == 0:
                origin = 'B'
                g1_x_offset = 0
                g1_y_offset = 1
                r_x_offset = 1
                r_y_offset = 1
                b_x_offset = 0
                b_y_offset = 0
                g2_x_offset = 1
                g2_y_offset = 0
            else:
                origin = 'G2'
                g1_x_offset = 1
                g1_y_offset = 1
                r_x_offset = 0
                r_y_offset = 1
                b_x_offset = 1
                b_y_offset = 0
                g2_x_offset = 0
                g2_y_offset = 0
    print(f"{origin=}")
    # fpb = np.zeros((width, height, 3), dtype=oiio.TypeUInt16)
    for r_y_ix in range(r_y_offset, height, 2):
        for r_x_ix in range(r_x_offset, width, 2):
            rgb_img[r_y_ix][r_x_ix][0] = pb[r_y_ix][r_x_ix]
    for g1_y_ix in range(g1_y_offset, height, 2):
        for g1_x_ix in range(g1_x_offset, width, 2):
            rgb_img[g1_y_ix][g1_x_ix][1] = pb[g1_y_ix][g1_x_ix]
    for g2_y_ix in range(g2_y_offset, height, 2):
        for g2_x_ix in range(g2_x_offset, width, 2):
            rgb_img[g2_y_ix][g2_x_ix][1] = pb[g2_y_ix][g2_x_ix]
    for b_y_ix in range(b_y_offset, height, 2):
        for b_x_ix in range(b_x_offset, width, 2):
            rgb_img[b_y_ix][b_x_ix][2] = pb[b_y_ix][b_x_ix]
    rgb_img = rgb_img / 65535.0
    output.write_image(rgb_img)
    output.close()


def offsets_for_anchor_in_pattern(anchor: Site, pattern: Pattern):
    g1_offs = None
    r_offs = None
    b_offs = None
    g2_offs = None
    if pattern == Pattern.GRBG:
        # G1  R G1  R
        #  B G2  B G2
        # G1  R G1  R
        if anchor == Site.G1:
            g1_offs = [0, 0]
            r_offs = [0, 1]
            b_offs = [1, 0]
            g2_offs = [1, 1]
        elif anchor == Site.R:
            g1_offs = [0, 1]
            r_offs = [0, 0]
            b_offs = [1, 1]
            g2_offs = [1, 0]
        elif anchor == Site.B:
            g1_offs = [1, 0]
            r_offs = [1, 1]
            b_offs = [0, 0]
            g2_offs = [0, 1]
        else:  # anchor is g2
            g1_offs = [1, 1]
            r_offs = [1, 0]
            b_offs = [0, 1]
            g2_offs = [0, 0]
    else:  # GBRG is only other supported pattern for now
        assert pattern == Pattern.GBRG
        # G1  B G1  B
        #  R G2  R G2
        # G1  B G1  B
        if anchor == Site.G1:
            g1_offs = [0, 0]
            r_offs = [0, 1]
            b_offs = [1, 0]
            g2_offs = [1, 1]
        elif anchor == Site.B:
            g1_offs = [1, 0]
            b_offs = [0, 0]
            r_offs = [1, 1]
            g2_offs = [0, 1]
        elif anchor == Site.R:
            g1_offs = [0, 1]
            b_offs = [1, 1]
            r_offs = [0, 0]
            g2_offs = [1, 0]
        else:  # anchor is g2
            g1_offs = [1, 1]
            r_offs = [1, 0]
            b_offs = [0, 1]
            g2_offs = [0, 0]
    return g1_offs, r_offs, b_offs, g2_offs


def check_exists_and_is_dir(dir_: Path, function):
    if not dir_.exists():
        raise FileNotFoundError(f"supplied {function} directory `{dir_}' does not exist.")
    if not dir_.is_dir():
        raise NotADirectoryError(f"supplied {function} directory `{dir_}' is not, in fact, a directory")


def input_sequence(args):
    # validate the input side
    if not args.input_dir:
        raise RuntimeError('an input directory must be specified')
    if not args.input_dir.endswith('/'):
        args.input_dir += '/'
    ari_dir = Path(args.input_dir)
    check_exists_and_is_dir(ari_dir, 'input')
    if not args.input_filename:
        raise RuntimeError('an input filename must be specified')
    wildcard = f"{args.input_dir}{args.input_filename}.@.ari"
    sequences = findSequencesOnDisk(wildcard)
    if not sequences:
        raise RuntimeError(f"indicated input directory contains no ARRIRAW (.ari) files with base filename "
                           f"`{args.input_filename}'")
    if len(sequences) > 1:
        raise RuntimeError(f"indicated input directory contains multiple sequences of ARRIRAW (.ari) files"
                           f" with base filename `{args.base_filename}'")
    sequence = sequences[0]
    first = args.first if args.first else sequence.start()
    if first < sequence.start():
        raise RuntimeError(f"indicated first frame {first} (inclusive bound) precedes actual first frame "
                           f"({sequence.start()})")
    last = args.last if args.last else sequence.end()
    if last > sequence.end() + 1:
        raise RuntimeError(f"indicated last frame {last} (exclusive bound) is beyond actual last frame "
                           f"({sequence.end()})")
    return sequence


def output_sequence(args, in_sequence):
    output_dir = Path(args.input_dir) if not args.output_dir else Path(args.output_dir)
    check_exists_and_is_dir(output_dir, 'output')
    sequence = [output_dir / f"{Path(frame).stem}.tif" for frame in in_sequence]
    return sequence


def main_loop(args, in_sequence, out_sequence):
    for in_path, out_path in zip(in_sequence, out_sequence):
        if out_path.exists():
            if not args.force:
                raise RuntimeError(f"file already exists at supplied output file path `{out_path}'")
        photosite_region = photosites_from_file(in_path, args.min_x, args.max_x, args.min_y, args.max_y, args.verbose)
        write_photosite_region_as_img(photosite_region, args.min_x, args.min_y, ALEXA_COLOR_ORDER, out_path)


def main(supplied_args):
    parser = ArgumentParser(description='Extract a rectangle of an ARRIRAW file and export as 16-bit RGB data')
    parser.add_argument('--input_dir', help='a directory where ARRIRAW files can be found')
    parser.add_argument('--input_filename', help='base filename of the ARRIRAW sequence, e.g. A016C001_210612_R26L')
    parser.add_argument('--output_dir', help='a directory where TIFF files into which the photosite data shall '
                        'be placed, with the components of the pixel not corresponding to the CFA color being zeroed, '
                        'shall be put')
    parser.add_argument('--min_x', type=int, help='leftmost column to be extracted (inclusive bound)')
    parser.add_argument('--max_x', type=int, help='rightmost column to be extracted (exclusive bound)')
    parser.add_argument('--min_y', type=int, help='bottommost row to be extracted (inclusive bound)')
    parser.add_argument('--max_y', type=int, help='topmost row to be extracted (exclusive bound)')
    parser.add_argument('--force', dest='force', default=False, action='store_true')
    parser.add_argument('--first', type=int, help='first frame number (inclusive bound)')
    parser.add_argument('--last', type=int, help='last frame number (exclusive bound)')
    parser.add_argument('--verbose', dest='verbose', default=False, action='store_true')
    parsed_args = parser.parse_args(supplied_args)
    in_sequence = input_sequence(parsed_args)
    out_sequence = output_sequence(parsed_args, in_sequence)
    main_loop(parsed_args, in_sequence, out_sequence)


if __name__ == '__main__':
    main(argv[1:])
