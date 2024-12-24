cd evaluation/pg19

MODELPATH=meta-llama/Llama-3.1-8B-Instruct
#lmsys/longchat-7b-v1.5-32k
OUTPUT_DIR=results/ppl_eval/Llama-3.1-8B-Instruct/amax
mkdir -p $OUTPUT_DIR

#device=0
budget=4096

CUDA_VISIBLE_DEVICES=0,1,2 python -u ppl_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens 6000 \
    #--quest --token_budget $budget --chunk_size 16 
    
#cd ..
#python get_repo_branch.py --model_path $MODELPATH