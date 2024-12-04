cd evaluation/passkey

MODEL=Llama-3.1-8B-Instruct
#Llama-3.2-1B-Instruct
#Yarn-Llama-2-7b-128k
MODELPATH=meta-llama/Llama-3.1-8B-Instruct
#meta-llama/Llama-3.2-1B-Instruct
#NousResearch/Yarn-Llama-2-7b-128k
OUTPUT_DIR=results/$MODEL/mean

mkdir -p $OUTPUT_DIR

length=100000

for token_budget in 256 512 1024 2048 4096
do
    python passkey.py -m $MODELPATH \
        --iterations 100 --fixed-length $length \
        --quest --token_budget $token_budget --chunk_size 16 \
        --output-file $OUTPUT_DIR/$MODEL-quest-$token_budget.jsonl
done


cd ..
python get_repo_branch.py --model_path $MODELPATH