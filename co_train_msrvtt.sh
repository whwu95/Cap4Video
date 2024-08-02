DATA_PATH=/data/Datasets/MSRVTT/MSRVTT_extract_frames/
python -m torch.distributed.launch --nproc_per_node=8  --master_port 2963 \
train_video.py \
--do_train  --num_thread_reader=4 --epochs=5 --batch_size=256 --n_display=20 \
--train_csv ${DATA_PATH}/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv data/MSRVTT_JSFUSION_test_titles.json  \
--data_path data/msrvtt_train_with_vitb32_max1_title_titles.json \
--features_path ${DATA_PATH}/video_extract_frames_30fps \
--output_dir ckpts/MSRVTT \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--strategy 2 \
--pretrained_clip_name ViT-B/32 \
--interaction wti  --text_pool_type transf_avg \
--world_size 8 \


python -m torch.distributed.launch --nproc_per_node=8  --master_port 2962 \
train_titles.py \
--do_train  --num_thread_reader=4 --epochs=5 --batch_size=256 --n_display=20 \
--train_csv ${DATA_PATH}/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv data/MSRVTT_JSFUSION_test_titles.json  \
--data_path data/msrvtt_train_with_vitb32_max1_title_titles.json \
--features_path ${DATA_PATH}/video_extract_frames_30fps \
--output_dir ckpts/MSRVTT \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt   --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--strategy 2 \
--pretrained_clip_name ViT-B/32 \
--interaction dp  --text_pool_type transf_avg \
--world_size 8 \


