#!/bin/sh

for GAME in zork1 sherlock
do
  export GAME
  for STARTING_PERCENTAGE in 0 20 40 60 80
  do
    export STARTING_PERCENTAGE
    sh train_ft.sh
  done
done
