import numpy as np
import pytest

from qat.runtime.ring_buffer import RingBuffer

SR = 100  # маленькая частота, чтобы считать сэмплы в уме


def ramp(start, n):
    """Сэмпл с номером i имеет значение i (float32 точно хранит целые до 2**24)."""
    return np.arange(start, start + n, dtype=np.float32)


def write_ramp(rb, total, block=7):
    for s in range(0, total, block):
        rb.write(ramp(s, min(block, total - s)))


def test_write_below_capacity():
    rb = RingBuffer(capacity_sec=10, sample_rate=SR)
    write_ramp(rb, 250)

    assert rb.available_range() == (0, 250)
    out = rb.extract(0, 250)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, ramp(0, 250))


def test_overflow_drops_oldest():
    rb = RingBuffer(capacity_sec=10, sample_rate=SR)  # 1000 сэмплов
    write_ramp(rb, 2345)

    assert rb.available_range() == (1345, 2345)
    np.testing.assert_array_equal(rb.extract(1345, 2345), ramp(1345, 1000))
    with pytest.raises(ValueError, match="вытеснены"):
        rb.extract(1344, 1400)


def test_extract_across_wrap_boundary():
    rb = RingBuffer(capacity_sec=10, sample_rate=SR)
    write_ramp(rb, 2100)

    # сэмплы 1990..2010 по модулю 1000 лежат на стыке конца и начала массива
    np.testing.assert_array_equal(rb.extract(1990, 2010), ramp(1990, 20))


def test_block_larger_than_capacity():
    rb = RingBuffer(capacity_sec=10, sample_rate=SR)
    rb.write(ramp(0, 123))
    rb.write(ramp(123, 2500))

    assert rb.available_range() == (1623, 2623)
    np.testing.assert_array_equal(rb.extract(1623, 2623), ramp(1623, 1000))


def test_clip_by_offsets():
    rb = RingBuffer(capacity_sec=30, sample_rate=SR)
    write_ramp(rb, 2000)

    clip = rb.clip(center=1200, start_offset_sec=-5.0, end_offset_sec=5.0)

    assert len(clip) == 10 * SR
    np.testing.assert_array_equal(clip, ramp(700, 1000))


def test_clip_default_offsets_at_16k():
    rb = RingBuffer()  # 30 с при 16 кГц
    rb.write(np.zeros(20 * 16000, dtype=np.float32))

    assert len(rb.clip(center=10 * 16000)) == 10 * 16000


def test_extract_future_raises():
    rb = RingBuffer(capacity_sec=10, sample_rate=SR)
    write_ramp(rb, 300)

    with pytest.raises(ValueError, match="ещё не записаны"):
        rb.clip(center=600)  # нужно с 100 до 1100, записано до 300


def test_clip_before_stream_start_raises():
    rb = RingBuffer(capacity_sec=10, sample_rate=SR)
    write_ramp(rb, 900)

    with pytest.raises(ValueError, match="до старта потока"):
        rb.clip(center=200)  # нужно с сэмпла -300


@pytest.mark.parametrize(
    "block",
    [np.zeros(10, dtype=np.float64), np.zeros((10, 2), dtype=np.float32)],
)
def test_write_rejects_wrong_format(block):
    rb = RingBuffer(capacity_sec=1, sample_rate=SR)
    with pytest.raises(ValueError, match="float32"):
        rb.write(block)
