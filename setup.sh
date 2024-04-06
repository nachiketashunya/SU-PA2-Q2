#!/bin/bash
set -eu  # Exit on error

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Define relative paths based on the script directory
storage_dir="$script_dir/data/raw"
extract_dir = "$script_dir/data/processed"
librispeech_dir="$extract_dir/LibriSpeech"
wham_dir="$extract_dir/wham_noise"
librimix_outdir="$extract_dir"

function LibriSpeech_test_clean() {
	if ! test -e $librispeech_dir/test-clean; then
		echo "Download LibriSpeech/test-clean into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
		tar -xzf $storage_dir/test-clean.tar.gz -C $extract_dir
		rm -rf $storage_dir/test-clean.tar.gz
	fi
}

function wham() {
	if ! test -e $wham_dir; then
		echo "Download wham_noise into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip -P $storage_dir
		unzip -qn $storage_dir/wham_noise.zip -d $extract_dir
		rm -rf $storage_dir/wham_noise.zip
	fi
}

LibriSpeech_test_clean &
wham
