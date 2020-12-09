DS=cao
set -e

run_exp () {
    DS=$1
    NAME=$2
    echo $DS and $NAME
    for msr in 1 4 7 9 11
    do
        python3 ../largecsrexp.py --dataset ${DS} --msr ${msr} --prior 0. > msr${msr}.p0.${NAME}.txt 2> msr${msr}.p0.${NAME}.log
    done
    for msr in 5 11 13 14 30
    do
        python3 ../largecsrexp.py --dataset ${DS} --msr $msr --prior 1 > msr${msr}.p1.${NAME}.txt 2> msr${msr}.p1.${NAME}.log
    done

}

run_exp cao4m cao4m
run_exp cao2m cao2m
run_exp 1.3M 1m
run_exp zeisel zeisel
