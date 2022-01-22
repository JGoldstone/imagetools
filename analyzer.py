from argparse import ArgumentParser
from pathlib import Path
import OpenImageIO as oiio

from input_sequence import InputSequence


class Analyzer(object):
    def __init__(self):
        self.parsed_args = None

    def oiio_coords(self):
        match self.parsed_args.app.lower():
            case ['nuke', 'arc']:
                return self.min_x, self.max_x, self.height - min_y, self.height - max_y
            case 'oiio':
                return self.min_x, self.max_x, self.min_y, self.max_y
            case _:
                raise RuntimeError(f"can't canonicalize coords because app `{app}' is unknown")

    def image_height(self, path: Path):
        if path.suffix == '.ari':
            return Arriraw.image_height(path)

    def main_loop(self, args, in_sequence, out_sequence):
        for in_path, out_path in zip(in_sequence, out_sequence):
            if out_path.exists():
                if not args.force:
                    raise RuntimeError(f"file already exists at supplied output file path `{out_path}'")
            photosite_region = photosites_from_file(in_path, args.min_x, args.max_x, args.min_y, args.max_y,
                                                    args.verbose)
            write_photosite_region_as_img(photosite_region, args.min_x, args.min_y, ALEXA_COLOR_ORDER, out_path)

    def main(self, supplied_args):
        parser = ArgumentParser(description='Extract a rectangle of an ARRIRAW file and export as 16-bit RGB data')
        parser.add_argument('--input_dir', help='a directory where ARRIRAW files can be found')
        parser.add_argument('--example_filename', help='example filename for the image sequence, e.g. '
                                                       ' A016C001_210612_R26L.1.ari')
        parser.add_argument('--output_dir', help='a directory where TIFF files into which the photosite data shall '
                                                 'be placed, with the components of the pixel not corresponding to the '
                                                 'CFA color being zeroed, shall be put')
        parser.add_argument('--app', help='application from which region coordinates come (determines location'
                            ' of coordinate system origin)')
        parser.add_argument('--min_x', type=int, help='leftmost column to be extracted (inclusive bound)')
        parser.add_argument('--max_x', type=int, help='rightmost column to be extracted (exclusive bound)')
        parser.add_argument('--min_y', type=int, help='bottommost row to be extracted (inclusive bound)')
        parser.add_argument('--max_y', type=int, help='topmost row to be extracted (exclusive bound)')
        parser.add_argument('--force', dest='force', default=False, action='store_true')
        parser.add_argument('--first', type=int, help='first frame number (inclusive bound)')
        parser.add_argument('--last', type=int, help='last frame number (exclusive bound)')
        parser.add_argument('--verbose', dest='verbose', default=False, action='store_true')
        self.parsed_args = parser.parse_args(supplied_args)
        in_sequence = InputSequence(self.parsed_args.input_dir, self.parsed_args.example_filename,
                                    self.parsed_args.first, self.parsed_args.last)
        oiio_min_x, oiio_max_x, oiio_min_y, oiio_max_y = self.oiio_coords_for(self.parsed_args.app,
                                                                              self.parsed_args.min_x,
                                                                              self.parsed_args.max_x,
                                                                              self.parsed_args.min_y,
                                                                              self.parsed_args.max_y)
        if Path(self.parsed_args.example_filename).suffix.lower() == '.ari':
            photosite_tiff_sequence = ARRIRAWExtractor(in_sequence,
                                                       self.parsed_args.min_x, self.parsed_args.max_x,
                                                       self.parsed_args.min_y, ).extracted_tiff_sequence()
            if not photosite_tiff_sequence:
                raise RuntimeError('ARRIRAW extractor did not return any extracted TIFF files')
            first_frame_path = Path(photosite_tiff_sequence[0])
            last_frame_path = Path(photosite_tiff_sequence[-1])
            extracted_dir = first_frame_path.parent
            extracted_filename = first_frame_path.name.split('.')[0]
            extracted_first_frame = first_frame_path.name.split('.')[1]
            extracted_last_frame = last_frame_path.name.split('.')[1]
            in_sequence = InputSequence(extracted_dir, extracted_filename, extracted_first_frame, extracted_last_frame)
        out_sequence = self.output_sequence(parsed_args, in_sequence)
        self.main_loop(parsed_args, in_sequence, out_sequence)

        if __name__ == '__main__':
            analyzer = Analyzer()
            analyzer.main(argv[1:])
