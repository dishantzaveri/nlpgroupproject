#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=2,3

GEN_SCRIPT_PATH=MileBench/generate.py
EVAL_SCRIPT_PATH=MileBench/evaluate.py
DATA_DIR=datasets/MileBench
MODEL_CONFIG_PATH=MileBench/configs/model_configs.yaml

gpu_num=2

for model in LLaVA_oneVision_Sink; do
    for dataset_name in ImageNeedleInAHaystack TextNeedleInAHaystack MMCoQA; do # GPR1200 ImageNeedleInAHaystack ObjectShuffle StateChange TextNeedleInAHaystack; do # ALFRED ActionLocalization ActionPrediction ActionSequence CLEVR-Change CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200 IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA TextNeedleInAHaystack WebQA WikiVQA nuscenes; do
        # Set batch size: max(int(batch_image/n_img),1)
        if [ ${dataset_name} = "MMCoQA" ] || [ ${dataset_name} = "NeedleInAHaystack" ] || [ ${dataset_name} = "GPR1200" ]
        then
            BATCH_SIZE=1
        else
            BATCH_SIZE=4
        fi
        echo "Batch size " $BATCH_SIZE

        mkdir -p logs/${model}_long
        # Start generating
        accelerate launch --config_file ./MileBench/configs/accelerate_configs.yaml \
            --main_process_port 29500  \
            --num_machines 1 \
            --machine_rank 0 \
            --num_processes ${gpu_num} \
            --deepspeed_multinode_launcher standard \
            \
            ${GEN_SCRIPT_PATH} \
            --data_dir ${DATA_DIR} \
            --dataset_name ${dataset_name}  \
            --model_name ${model} \
            --output_dir outputs \
            --batch-image ${BATCH_SIZE} \
            --model_configs ${MODEL_CONFIG_PATH} \
            --overwrite \
            > logs/${model}_long/${dataset_name}.log

        # Start evaluating
        python ${EVAL_SCRIPT_PATH} \
            --data-dir ${DATA_DIR} \
            --dataset ${dataset_name} \
            --result-dir outputs/${model}_long \
            >> logs/${model}_long/${dataset_name}.log

        # ############################## Combined to 1 image ###########################
        # # Start generating
        # accelerate launch --config_file ./configs/accelerate_configs.yaml \
        #     --main_process_port 29500  \
        #     --num_machines 1 \
        #     --machine_rank 0 \
        #     --num_processes ${gpu_num}  \
        #     --deepspeed_multinode_launcher standard \
        #     \
        #     ${GEN_SCRIPT_PATH} \
        #     --data_dir ${DATA_DIR} \
        #     --dataset_name ${dataset_name}  \
        #     --model_name ${model} \
        #     --output_dir outputs_combine_1 \
        #     --batch-image ${BATCH_SIZE} \
        #     --model_configs ${MODEL_CONFIG_PATH} \
        #     --overwrite \
        #     --combine_image 1 \
        #     > logs/${model}/${dataset_name}_combine_1.log

        # # Start evaluating
        # python ${EVAL_SCRIPT_PATH} \
        #     --data-dir ${DATA_DIR} \
        #     --dataset ${dataset_name} \
        #     --result-dir outputs_combine_1/${model} \
        #     >> logs/${model}/${dataset_name}_combine_1.log
    done
done
