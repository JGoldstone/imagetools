from pathlib import Path
from fileseq import findSequenceOnDisk

class InputSequence(object):
    @staticmethod
    def check_exists_and_is_dir(dir_: Path, dir_desc):
        if not dir_.exists():
            raise FileNotFoundError(f"supplied {dir_desc} directory `{dir_}' does not exist.")
        if not dir_.is_dir():
            raise NotADirectoryError(f"supplied {dir_desc} directory `{dir_}' is not, in fact, a directory")

    def __init__(self, dir_, file_, first_=None, last_=None):
        # validate the input side
        if not dir_:
            raise RuntimeError('no input directory was specified')
        if not dir_.endswith('/'):
            raise RuntimeError(f"specified input directory `{dir_}' does not end with `/'")
        InputSequence.check_exists_and_is_dir(Path(dir_), 'input')
        self.directory = dir_
        if not file_:
            raise RuntimeError('no input filename was specified')
        self.filename = file_
        wildcard = f"{self.directory}{self.filename}.@.{Path(self.filename).suffix}"
        sequence = findSequenceOnDisk(wildcard)
        if not sequence:
            raise RuntimeError(f"specified input directory contains no files matching pattern `{wildcard}'")
        if first_ and first_ < sequence.start():
            raise RuntimeError(f"indicated first frame {first_} (inclusive bound) precedes actual first frame "
                               f"({sequence.start()})")
        self.first = first_ if first_ else sequence.start()
        if not last_:
            self.last = sequence.end()
        else:
            if last_ > sequence.end() + 1:
                raise RuntimeError(f"indicated last frame {last_} (exclusive bound) is beyond actual last frame "
                                   f"({sequence.end()})")

    def output_sequence(self, args, in_sequence):
        output_dir = Path(args.input_dir) if not args.output_dir else Path(args.output_dir)
        InputSequence.check_exists_and_is_dir(output_dir, 'output')
        sequence = [output_dir / f"{Path(frame).stem}.tif" for frame in in_sequence]
        return sequence
