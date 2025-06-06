# FishFry
Unpacking and processing scripts for the FishStand app, organized by `Analysis`. For `TriggeredImage` scripts, see the repository `swaneyjr/beamtools` instead.

## Calibration

Pixel masking and lens-shade scaling can be accomplished through `pixelstats` and `cosmics`: see [our paper](http://arXiv.org/abs/2107.06332) for the approach. Results are saved in and accessed from a directory specified by the `--calib` flag. To export these calibrations to be read by the `FishStand` app, run:

```
./pixelstats/export.py --calib path/to/calib/dir/
```

and copy the resulting `.cal` files to the base `FishStand/` directory on the Android device. To acquire lens-shading corrections, the scripts `pixelstats/gain.py`, `pixelstats/lens_shading.py`, and `pixelstats/electrons.py` are run in series. To update a set of calibrations with additional hot pixels found during further `cosmics` runs, the output of `cosmics/hot.py` can be used in conjunction with `pixelstats/add_online_hotcells.py`. 