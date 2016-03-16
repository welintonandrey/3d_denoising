#!/bin/bash

# read the options
TEMP=`getopt -o f:s:i:m:h --long folder:,sigma:,images:,msb:,help -n 'runAux.sh' -- "$@"`
eval set -- "$TEMP"

function PrintHelp() {
  echo "Parameters:"
  echo "    -f | --folder : base directory"
  echo "    -s | --sigma  : sigma value for the gaussian noise"
  echo "    -i | --images : frame sequence folder (relative path)"
  echo "    -m | --msb    : number of MSB to be considered"
  exit 1
}

if [ \( "$#" -lt 9 \) -a \( "$#" -ne 2 \) ]
then
	echo -e "Missing pararameters! \nSee -h or --help";
	exit 1
else
	while true ; do

		case "$1" in
		    -f|--folder)
		        case "$2" in
		            "") shift 2 ;;
		            *) FOLDER=$2 ; shift 2 ;;
		        esac ;;
		    -s|--sigma)
		        case "$2" in
		            "") shift 2 ;;
		            *) SIGMA=$2 ; shift 2 ;;
		        esac ;;
		    -i|--images)
		        case "$2" in
		            "") shift 2 ;;
		            *) IMAGES=$2 ; shift 2 ;;
		        esac ;;
		    -m|--msb)
		        case "$2" in
		            "") shift 2 ;;
		            *) MSB=$2 ; shift 2 ;;
		        esac ;;
			-h|--help) PrintHelp ;;
		    --) shift ; break ;;
		    *) echo "Internal error!" ; exit 1 ;;
		esac
	done
fi


for h in {10,15,20,25}
do
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-p3-w5-h$h/ -ori $FOLDER$IMAGES/original/ -H $h -p 3 -w 5 -sig $SIGMA -msb $MSB -seq $IMAGES -f $FOLDER
done

for w in {7,9,11,13}
do
  for h in {10,15,20,25}
  do
    ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-p5-w$w-h$h/ -ori $FOLDER$IMAGES/original/ -H $h -p 5 -w $w -sig $SIGMA -msb $MSB -seq $IMAGES -f $FOLDER
  done
done
