@echo off
set GDB9_URL=https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
set QM9_CSV_URL=https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
set QM9_TASKS=mu,alpha,homo,lumo,gap,r2,zpve,cv,u0,u298,h298

echo Downloading gdb9.tar.gz...
curl %GDB9_URL% -o gdb9.tar.gz

echo Extracting gdb9.sdf...
tar -xzvf gdb9.tar.gz gdb9.sdf

echo Converting gdb9.sdf to gdb9.sdf.csv...
obabel -isdf gdb9.sdf -ocsv -h -xS -p %QM9_TASKS% -O gdb9.sdf.csv

echo Downloading qm9.csv...
curl %QM9_CSV_URL% -o qm9.csv

echo Done!