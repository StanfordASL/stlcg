#!/bin/zsh

echo "stlcg settling"
{time python3 pstl.py 1 1 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 10 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 100 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 200 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 400 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 500 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 600 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 700 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 800 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 900 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 1 1000 10 ; } &>> compute_time_raw.txt


echo "binary search settling"
{time python3 pstl.py 2 1 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 10 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 100 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 200 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 400 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 500 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 600 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 700 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 800 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 900 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 2 1000 10 ; } &>> compute_time_raw.txt

echo "stlcg overshoot"
{time python3 pstl.py 3 1 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 10 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 100 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 200 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 400 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 500 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 600 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 700 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 800 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 900 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 3 1000 10 ; } &>> compute_time_raw.txt

echo "binary search overshoot"
{time python3 pstl.py 4 1 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 10 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 100 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 200 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 400 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 500 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 600 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 700 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 800 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 900 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 4 1000 10 ; } &>> compute_time_raw.txt

echo "stlcg-gpu settling"
{time python3 pstl.py 5 1 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 10 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 100 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 200 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 400 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 500 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 600 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 700 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 800 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 900 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 5 1000 10 ; } &>> compute_time_raw.txt

echo "stlcg-gpu overshoot"
{time python3 pstl.py 6 1 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 10 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 100 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 200 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 400 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 500 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 600 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 700 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 800 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 900 10 ; } &>> compute_time_raw.txt
{time python3 pstl.py 6 1000 10 ; } &>> compute_time_raw.txt