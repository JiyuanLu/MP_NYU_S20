#!/bin/sh
max=12
n=10
for (( i = 1; i <= $max; ++i ))
do
    echo "$i"
    for (( j = 0; j < $n; ++j ))
    do
        echo "$(time ./ptsm $i $OMP_NUM_THREADS "cities$i.txt")"
    done
done

