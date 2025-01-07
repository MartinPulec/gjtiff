#!/bin/sh

handle_error() {
  printf "HTTP/1.1 500 Internal Server Error\r\nContent-Type: \
text/plain\r\n\r\n"
  echo "LOG:"
  cat "$log"
  rm "$input_filename" "$log"
  exit 0
}

log=$(mktemp)
exec 3>&2
exec 2>"$log"

set | grep -E '^(CONTENT|HTTP)' >&3

read -r boundary
boundary=$(echo "$boundary" | tr -d '\r')

while read -r n; [ -n "$(echo "$n" | tr -d '\n\r')" ]
do
   if echo "$n" | grep -q filename=; then
     input_filename=$(echo "$n" |
      sed 's/^Content-Disposition.*filename=\"\([^""]*\)\".*$/\1/')
   fi
done

if [ -z "$input_filename" ] ||
    echo "$input_filename" | grep -q '/'; then
  echo "Empty or invalid filename: $input_filename" > "$log"
  handle_error
fi

echo "input file: $input_filename" >&3

cat > "$input_filename"
off=$(strings -t d "$input_filename" | grep -- "$boundary" | awk '{ print $1 }')
truncate -s $((off - 2)) "$input_filename" # drop \r\n

exec 2>&3

if [ -s "$log" ]; then
  handle_error
fi

gjtiff -vvv "$input_filename" >"$log" 2>&1
GJ_RC=$?

cat "$log" >&2

if [ $GJ_RC -ne 0 ]; then
  handle_error
fi

output_filename=${input_filename%.*}.jpg
printf "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n"
printf "Content-Disposition: inline; filename=\"%s\"\r\n" "$output_filename"
printf "\r\n"
cat "$output_filename"
echo "Sending output file: $output_filename" >&2

rm "$output_filename"
rm "$input_filename" "$log"
