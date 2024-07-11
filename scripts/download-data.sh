# Initialize an empty array to store the second words
declare -a second_words

# Read the file line by line
while IFS= read -r line; do
	    # Split the line into words
	        read -ra words <<< "$line"
    
    # Check if there's a second word and append it to the array
    if [ ${#words[@]} -gt 1 ]; then
        second_words+=("${words[1]}")
    fi
done < data/nasdaq_tickers.txt

# Sort the array
IFS=$'\n' sorted_words=($(sort <<< "${second_words[*]}"))

# Echo each sorted element
echo "Downloading in Parallel:"
max_jobs=${2:-50}
current_jobs=0
for word in "${sorted_words[@]}"; do
    while (( current_jobs >= max_jobs )); do
        wait -n
        ((current_jobs--))
    done
    python3 scripts/download_ticker.py $word $1 &
    
    ((current_jobs++))
done
