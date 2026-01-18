#before using, make sure the fixloss in core.py is multilayer

python drag_multi.py --model checkpoints/stylegan2-ffhq-512x512.pkl --dataset FreeDragBench/FreeDragBench/ --category faces --out drag_multilayer --max-samples 10 --iterations 40 --fixloss-type multilayer

python eval_batch.py --root ./drag_multilayer  --category faces --recursive