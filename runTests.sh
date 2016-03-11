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

if [ "$SIGMA" -le 15 ]
then
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h10-p3-w5/ -ori $FOLDER$IMAGES/original/ -H 10 -p 3 -w 5 -sig $SIGMA -msb $MSB;
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h15-p3-w5/ -ori $FOLDER$IMAGES/original/ -H 15 -p 3 -w 5 -sig $SIGMA -msb $MSB;
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h20-p3-w5/ -ori $FOLDER$IMAGES/original/ -H 20 -p 3 -w 5 -sig $SIGMA -msb $MSB;
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h25-p3-w5/ -ori $FOLDER$IMAGES/original/ -H 25 -p 3 -w 5 -sig $SIGMA -msb $MSB;
fi

for w in {5,7,9,13}
do
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h10-p5-w$w/ -ori $FOLDER$IMAGES/original/ -H 10 -p 5 -w $w -sig $SIGMA -msb $MSB;
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h15-p5-w$w/ -ori $FOLDER$IMAGES/original/ -H 15 -p 5 -w $w -sig $SIGMA -msb $MSB;
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h20-p5-w$w/ -ori $FOLDER$IMAGES/original/ -H 20 -p 5 -w $w -sig $SIGMA -msb $MSB;
  ./runNonLocalMeans -in $FOLDER$IMAGES/noise$SIGMA/ -out  $FOLDER$IMAGES/res/sigma$SIGMA/res-sigma$SIGMA-h25-p5-w$w/ -ori $FOLDER$IMAGES/original/ -H 25 -p 5 -w $w -sig $SIGMA -msb $MSB;
done
