# Cleaning DataSet CMD using perl & bash CMD

## Line Count
```bash
wc -l <filename>
```

## file ထဲက sentencs တွေထဲမှာ format မမှန်တာတွေကို စစ်မယ်
```bash
awk -F',' 'NF != 2' emotions_mya1.csv | wc -l ## check [,]
grep -n '["]' emotions_mya1.csv | wc -l  ## check ["]
```
## Remove Blank Sentences
```bash
awk -F',' '$1 != ""' emotions_mya1.csv > step1.csv
```

## Clean " and make one sentence
```bash
perl -CSDA -0777 -pe 's#"([^"]*)",(\d+)#do { my $t=$1; $t =~ s/\R/ /g; "$t,$2" }#ge' <filename> > <new_filename> ##error
perl -CSDA -0777 -pe 's#"([^"]*)",(.*?)#do { my $t=$1; $t =~ s/\R/ /g; "$t,$2" }#ge' step1.csv > step2.csv
```

## Check Duplicate Sentences
file ထဲမှာ တူညီတဲ့ စာကြောင်းတွေကို စစ်ဆေးဖို့ 
```bash
sort step2.csv | uniq -d
```
file ထဲမှာ တူညီတဲ့ စာကြောင်းတွေကို အရေတွက် ဘယ်လောက်ပါလဲ စစ်ဆေးရန်
```bash
sort step2.csv | uniq -c | sort -nr | head
```

## Cut Duplicate Sentences
```bash 
sort step2.csv | uniq > step3.csv
```

### Options
## sentences တွေမှာ ၊ / ။ ပါမပါစစ်ဆေးရန် လိုမလို
``` bash
grep -n '[၊။"]' step3.csv | wc -l
sed 's/[၊။]//g' <step3.csv> > <delete.csv>
```

### Double Check 
## sentences ထဲမှာ " ထပ်ပါနေသေးလို့ ပြန်စစ်ဆေးပြီးဖျက်ထား
```bash
grep -n '["]' delete.csv | wc -l
sed 's/"//g' delete.csv > final.csv
```

## Data clear လုပ်တဲ့ချိန် sorted လုပ်လိုက်လို့ shuffle ပြန်လုပ်သင့်
``` bash
(head -n 1 final.csv && tail -n +2 final.csv | shuf) > shuffled.csv
``` 

## Train & Test dataset ခွဲလိုလျှင်
```bash
head -n 1000 shuffled.csv > train.csv
tail -n +1001 shuffled.csv > test.csv
```