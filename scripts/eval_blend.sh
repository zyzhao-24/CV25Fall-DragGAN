#before using, make sure the fixloss in core.py is multilayer

python drag_multi.py --model checkpoints/stylegan2-ffhq-512x512.pkl --dataset FreeDragBench/FreeDragBench/ --category faces --out drag_blend --max-samples 10 --iterations 40 --fixloss-type blended

python eval_batch.py --root ./drag_blend  --category faces --recursive