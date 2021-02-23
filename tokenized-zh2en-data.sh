set -e

N_THREADS=16    # number of threads in data preprocessing

#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --prep)
    PREP="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"


#
# Check parameters
#
if [ "$SRC" == "" ]; then echo "--src not provided"; exit; fi
if [ "$TGT" == "" ]; then echo "--tgt not provided"; exit; fi
if [ "$SRC" == "$TGT" ]; then echo "source and target cannot be identical"; exit; fi
if [ "$PREP" == "" ]; then echo "--prep not provided"; exit; fi

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
TMP=$PREP/tmp

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $TMP

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl


# valid / test file raw data
unset PARA_SRC_TRAIN PARA_TGT_TRAIN PARA_SRC_VALID PARA_TGT_VALID PARA_SRC_TEST PARA_TGT_TEST    # Update

PARA_SRC_TRAIN_RAW=$PREP/train.$SRC
PARA_TGT_TRAIN_RAW=$PREP/train.$TGT 
PARA_SRC_VALID_RAW=$PREP/valid.$SRC
PARA_TGT_VALID_RAW=$PREP/valid.$TGT
PARA_SRC_TEST_RAW=$PREP/test.$SRC
PARA_TGT_TEST_RAW=$PREP/test.$TGT

PARA_SRC_TRAIN=$TMP/train.$SRC.tok
PARA_TGT_TRAIN=$TMP/train.$TGT.tok 
PARA_SRC_VALID=$TMP/valid.$SRC.tok
PARA_TGT_VALID=$TMP/valid.$TGT.tok
PARA_SRC_TEST=$TMP/test.$SRC.tok
PARA_TGT_TEST=$TMP/test.$TGT.tok

# preprocessing commands - special case for Romanian
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"

# check valid and test files are here
if ! [[ -f "$PARA_SRC_TRAIN_RAW" ]]; then echo "$PARA_SRC_TRAIN_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_TGT_TRAIN_RAW" ]]; then echo "$PARA_TGT_TRAIN_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_SRC_VALID_RAW" ]]; then echo "$PARA_SRC_VALID_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_TGT_VALID_RAW" ]]; then echo "$PARA_TGT_VALID_RAW is not found!"; exit; fi
if ! [[ -f "$PARA_SRC_TEST_RAW" ]];  then echo "$PARA_SRC_TEST_RAW is not found!";  exit; fi
if ! [[ -f "$PARA_TGT_TEST_RAW" ]];  then echo "$PARA_TGT_TEST_RAW is not found!";  exit; fi

echo "Tokenizing Chinese data..."
python -m jieba $PARA_SRC_TRAIN_RAW -d > $PARA_SRC_TRAIN
python -m jieba $PARA_SRC_VALID_RAW -d > $PARA_SRC_VALID
python -m jieba $PARA_SRC_TEST_RAW -d > $PARA_SRC_TEST

echo "Tokenizing English data..."
eval "cat $PARA_TGT_TRAIN_RAW | $TGT_PREPROCESSING > $PARA_TGT_TRAIN"
eval "cat $PARA_TGT_VALID_RAW | $TGT_PREPROCESSING > $PARA_TGT_VALID"
eval "cat $PARA_TGT_TEST_RAW | $TGT_PREPROCESSING > $PARA_TGT_TEST"