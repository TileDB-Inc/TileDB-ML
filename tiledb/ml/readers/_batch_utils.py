from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True, repr=False)
class Batch:
    size: int
    x_read_slice: Optional[slice]
    y_read_slice: Optional[slice]
    x_buffer_slice: slice
    y_buffer_slice: slice

    def __repr__(self) -> str:
        x, y = self.x_buffer_slice, self.y_buffer_slice
        s = f"Batch({self.size}, x[{x.start}:{x.stop}], y[{y.start}:{y.stop}]"
        if self.x_read_slice:
            s += f", x_read[{self.x_read_slice.start}:{self.x_read_slice.stop}]"
        if self.y_read_slice:
            s += f", y_read[{self.y_read_slice.start}:{self.y_read_slice.stop}]"
        return s + ")"


def iter_batches(
    x_buffer_size: int,
    y_buffer_size: int,
    start_offset: int,
    stop_offset: int,
) -> Iterator[Batch]:
    """
    Generate `Batch` instances describing each batch.

    Each yielded `Batch` instance describes:
    - Its size
    - The slice to read from the x array (if the current x buffer is consumed).
    - The slice to read from the y array (if the current y buffer is consumed).
    - The batch slice to read from the x buffer.
    - The batch slice to read from the y buffer.

    :param x_buffer_size: (Max) size of the x buffer.
    :param y_buffer_size: (Max) size of the y buffer.
    :param start_offset: Start row offset.
    :param stop_offset: Stop row offset.
    """
    x_read_slices = iter_slices(start_offset, stop_offset, x_buffer_size)
    y_read_slices = iter_slices(start_offset, stop_offset, y_buffer_size)
    x_buf_offset = x_buffer_size
    y_buf_offset = y_buffer_size
    offset = start_offset
    while offset < stop_offset:
        if x_buf_offset == x_buffer_size:
            x_read_slice = next(x_read_slices)
            x_read_size = x_read_slice.stop - x_read_slice.start
            x_buf_offset = 0
        else:
            x_read_slice = None

        if y_buf_offset == y_buffer_size:
            y_read_slice = next(y_read_slices)
            y_read_size = y_read_slice.stop - y_read_slice.start
            y_buf_offset = 0
        else:
            y_read_slice = None

        buffer_size = min(x_read_size - x_buf_offset, y_read_size - y_buf_offset)
        x_next_buf_offset = x_buf_offset + buffer_size
        y_next_buf_offset = y_buf_offset + buffer_size
        yield Batch(
            buffer_size,
            x_read_slice,
            y_read_slice,
            slice(x_buf_offset, x_next_buf_offset),
            slice(y_buf_offset, y_next_buf_offset),
        )
        x_buf_offset = x_next_buf_offset
        y_buf_offset = y_next_buf_offset
        offset += buffer_size


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    offsets = range(start, stop, step)
    yield from map(slice, offsets, offsets[1:])
    yield slice(offsets[-1], stop)
