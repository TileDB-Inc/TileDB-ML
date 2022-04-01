from tiledb.ml.readers._batch_utils import iter_batches


def test_iter_batches():
    batches = iter_batches(
        x_buffer_size=17, y_buffer_size=41, start_offset=104, stop_offset=213
    )
    assert list(map(str, batches)) == [
        "Batch(17, x[0:17], y[0:17], x_read[104:121], y_read[104:145])",
        "Batch(17, x[0:17], y[17:34], x_read[121:138])",
        "Batch(7, x[0:7], y[34:41], x_read[138:155])",
        "Batch(10, x[7:17], y[0:10], y_read[145:186])",
        "Batch(17, x[0:17], y[10:27], x_read[155:172])",
        "Batch(14, x[0:14], y[27:41], x_read[172:189])",
        "Batch(3, x[14:17], y[0:3], y_read[186:213])",
        "Batch(17, x[0:17], y[3:20], x_read[189:206])",
        "Batch(7, x[0:7], y[20:27], x_read[206:213])",
    ]
    batches = iter_batches(
        x_buffer_size=18, y_buffer_size=27, start_offset=104, stop_offset=213
    )
    assert list(map(str, batches)) == [
        "Batch(18, x[0:18], y[0:18], x_read[104:122], y_read[104:131])",
        "Batch(9, x[0:9], y[18:27], x_read[122:140])",
        "Batch(9, x[9:18], y[0:9], y_read[131:158])",
        "Batch(18, x[0:18], y[9:27], x_read[140:158])",
        "Batch(18, x[0:18], y[0:18], x_read[158:176], y_read[158:185])",
        "Batch(9, x[0:9], y[18:27], x_read[176:194])",
        "Batch(9, x[9:18], y[0:9], y_read[185:212])",
        "Batch(18, x[0:18], y[9:27], x_read[194:212])",
        "Batch(1, x[0:1], y[0:1], x_read[212:213], y_read[212:213])",
    ]
