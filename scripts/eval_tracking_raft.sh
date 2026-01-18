#before using, make sure the fixloss in core.py is single

python drag_multi.py --model checkpoints/stylegan2-ffhq-512x512.pkl --dataset FreeDragBench/FreeDragBench/ --category faces --out drag_tracking_raft --max-samples 10 --iterations 40 --fixloss-type single --tracking-method raft

python eval_batch.py --root ./drag_tracking_raft  --category faces --recursive