#! /bin/bash
source activate EDNE

mkdir bash/logg
mkdir bash/logg_1
mkdir bash/logg_2
mkdir bash/logg_3
mkdir bash/logg_4
mkdir bash/logg_5
mkdir bash/logg_6
mkdir bash/logg_7
mkdir bash/logg_8
mkdir bash/logg_9
mkdir bash/logg_10

echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_1
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_1
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_1
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_1
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_1









echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_2
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_2
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_2
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_2
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_2







echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_3
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_3
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_3
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_3
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_3







echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_4
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_4
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_4
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_4
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_4






echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_5
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_5
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_5
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_5
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_5







echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_6
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_6
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_6
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_6
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_6





echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_7
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_7
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_7
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_7
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_7






echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_8
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_8
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_8
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_8
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_8





echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_9
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_9
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_9
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_9
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_9




echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_20.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_20.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_20.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_40.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_40.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_40.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_60.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_60.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_60.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_80.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_80.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_80.pkl
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt --------
echo -------- SG-EDNE_m5_r1_s1_DNC-Email_100.txt -------- > bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
start=`date +%s`
python src/main.py --method SG-EDNE --task save --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --num-walks 10 --walk-length 80 --window 10 --negative 5 --seed 2021 --workers 16 --emb-dim 128 --num-base-models 5 --restart 1 --max-r-prob 0.1 --scaling_strategy 1  >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
end=`date +%s`
echo --- ALL embedding time: $((end-start)) ---
echo ALL embedding time: $((end-start)) >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
echo --- start eval downstream tasks ---
python src/eval.py --task all --graph data/DNC-Email_100.pkl --emb-file output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl --seed 2021 >> bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt
rm output/emb_SG-EDNE_m5_r1_s1_DNC-Email_100.pkl

mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_20.txt bash/logg_10
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_40.txt bash/logg_10
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_60.txt bash/logg_10
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_80.txt bash/logg_10
mv bash/logg/SG-EDNE_m5_r1_s1_DNC-Email_100.txt bash/logg_10