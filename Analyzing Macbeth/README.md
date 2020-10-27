# Project: Analyzing Macbeth Using NLP



```python
import requests
macbeth = requests.get('http://www.gutenberg.org/cache/epub/2264/pg2264.txt').text

print(type(macbeth))
print(len(macbeth))
print(macbeth[:500])
```

    <class 'str'>
    120253
    ﻿
    
    ***The Project Gutenberg's Etext of Shakespeare's First Folio***
    ********************The Tragedie of Macbeth*********************
    
    
    
    *******************************************************************
    THIS EBOOK WAS ONE OF PROJECT GUTENBERG'S EARLY FILES PRODUCED AT A
    TIME WHEN PROOFING METHODS AND TOOLS WERE NOT WELL DEVELOPED. THERE
    IS AN IMPROVED EDITION OF THIS TITLE WHICH MAY BE VIEWED AS EBOOK
    (#1533) at https://www.gutenberg.org/ebooks/1533
    *********************************



```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

words = macbeth.split()
word_counts = {}
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1
    
counts = list(word_counts.items())
top_25 = sorted(counts, key = lambda x: x[1], reverse=True)[:25]
y = [item[1] for item in top_25]
X = np.arange(len(y))
plt.figure(figsize=(13,13))
plt.bar(X , y)
plt.xticks(X, [item[0] for item in top_25]);
plt.ylabel('Number of Occurences')
plt.xlabel('Word')
plt.title('Top 25 Words in Macbeth')
```




    Text(0.5, 1.0, 'Top 25 Words in Macbeth')




![png](output_2_1.png)



```python
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk import FreqDist
from nltk import word_tokenize
import string
import re
```


```python
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
macbeth_tokens_raw = nltk.regexp_tokenize(macbeth, pattern)
macbeth_tokens = [word.lower() for word in macbeth_tokens_raw]
```


```python
macbeth_freqdist = FreqDist(macbeth_tokens)
macbeth_freqdist.most_common(50)
```




    [('the', 764),
     ('and', 603),
     ('to', 460),
     ('of', 428),
     ('i', 344),
     ('a', 287),
     ('you', 269),
     ('that', 245),
     ('in', 225),
     ('is', 213),
     ('my', 207),
     ('it', 185),
     ('not', 182),
     ('with', 162),
     ('this', 159),
     ('be', 153),
     ('his', 147),
     ('for', 139),
     ('your', 139),
     ('macb', 137),
     ('our', 136),
     ('but', 126),
     ('haue', 122),
     ('me', 115),
     ('all', 112),
     ('he', 112),
     ('what', 110),
     ('as', 109),
     ('so', 108),
     ('we', 100),
     ('him', 92),
     ('are', 89),
     ('thou', 87),
     ('or', 85),
     ('which', 83),
     ('enter', 81),
     ('will', 80),
     ('they', 79),
     ('by', 74),
     ('no', 73),
     ('from', 71),
     ('on', 70),
     ('if', 68),
     ('shall', 68),
     ('macbeth', 67),
     ('then', 67),
     ('at', 66),
     ('their', 62),
     ('thee', 61),
     ('more', 58)]




```python
stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)
stopwords_list += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'project', 'gutenberg']

macbeth_words_stopped = [word for word in macbeth_tokens if word not in stopwords_list]
```


```python
macbeth_stopped_freqdist = FreqDist(macbeth_words_stopped)
macbeth_stopped_freqdist.most_common(50)
```




    [('macb', 137),
     ('haue', 122),
     ('thou', 87),
     ('enter', 81),
     ('shall', 68),
     ('macbeth', 67),
     ('thee', 61),
     ('vpon', 58),
     ('macd', 58),
     ('yet', 57),
     ('come', 56),
     ('thy', 56),
     ('king', 55),
     ('vs', 55),
     ('time', 54),
     ('hath', 52),
     ('may', 51),
     ('good', 50),
     ('rosse', 49),
     ('would', 48),
     ('lady', 48),
     ('like', 45),
     ('one', 44),
     ('make', 39),
     ('say', 39),
     ('must', 38),
     ('doe', 38),
     ('lord', 38),
     ('see', 37),
     ('tis', 37),
     ('selfe', 36),
     ('etext', 35),
     ('done', 35),
     ('ile', 35),
     ('feare', 35),
     ('let', 35),
     ('well', 34),
     ('know', 34),
     ('man', 34),
     ('wife', 34),
     ('night', 34),
     ('banquo', 34),
     ('great', 32),
     ('exeunt', 30),
     ('speake', 29),
     ('sir', 29),
     ('lenox', 28),
     ('things', 27),
     ('mine', 26),
     ('vp', 26)]




```python
len(macbeth_stopped_freqdist)
```




    3981




```python
total_word_count = sum(macbeth_stopped_freqdist.values())
macbeth_top_50 = macbeth_stopped_freqdist.most_common(50)
print('Word\t\t\tNormalized Frequency')
for word in macbeth_top_50:
    normalized_frequency = word[1] / total_word_count
    print('{} \t\t\t {:.4}'.format(word[0], normalized_frequency))
```

    Word			Normalized Frequency
    macb 			 0.01196
    haue 			 0.01065
    thou 			 0.007597
    enter 			 0.007073
    shall 			 0.005938
    macbeth 			 0.005851
    thee 			 0.005327
    vpon 			 0.005065
    macd 			 0.005065
    yet 			 0.004977
    come 			 0.00489
    thy 			 0.00489
    king 			 0.004803
    vs 			 0.004803
    time 			 0.004715
    hath 			 0.004541
    may 			 0.004453
    good 			 0.004366
    rosse 			 0.004279
    would 			 0.004191
    lady 			 0.004191
    like 			 0.003929
    one 			 0.003842
    make 			 0.003406
    say 			 0.003406
    must 			 0.003318
    doe 			 0.003318
    lord 			 0.003318
    see 			 0.003231
    tis 			 0.003231
    selfe 			 0.003144
    etext 			 0.003056
    done 			 0.003056
    ile 			 0.003056
    feare 			 0.003056
    let 			 0.003056
    well 			 0.002969
    know 			 0.002969
    man 			 0.002969
    wife 			 0.002969
    night 			 0.002969
    banquo 			 0.002969
    great 			 0.002794
    exeunt 			 0.00262
    speake 			 0.002532
    sir 			 0.002532
    lenox 			 0.002445
    things 			 0.002358
    mine 			 0.00227
    vp 			 0.00227



```python
bigram_measures = nltk.collocations.BigramAssocMeasures()
```


```python
macbeth_finder = BigramCollocationFinder.from_words(macbeth_words_stopped)
```


```python
macbeth_scored = macbeth_finder.score_ngrams(bigram_measures.raw_freq)
```


```python
macbeth_scored[:50]
```




    [(('enter', 'macbeth'), 0.0013971358714634998),
     (('exeunt', 'scena'), 0.001309814879497031),
     (('thane', 'cawdor'), 0.0011351728955640936),
     (('knock', 'knock'), 0.0008732099196646874),
     (('lord', 'macb'), 0.0007858889276982187),
     (('thou', 'art'), 0.0007858889276982187),
     (('good', 'lord'), 0.0006985679357317499),
     (('haue', 'done'), 0.0006985679357317499),
     (('macb', 'haue'), 0.0006985679357317499),
     (('small', 'print'), 0.0006985679357317499),
     (('enter', 'lady'), 0.0006112469437652812),
     (('first', 'folio'), 0.0006112469437652812),
     (('let', 'vs'), 0.0006112469437652812),
     (('tragedie', 'macbeth'), 0.0006112469437652812),
     (('macbeth', 'macb'), 0.0005239259517988125),
     (('public', 'domain'), 0.0005239259517988125),
     (('carnegie', 'mellon'), 0.0004366049598323437),
     (('enter', 'malcolme'), 0.0004366049598323437),
     (('enter', 'three'), 0.0004366049598323437),
     (('euery', 'one'), 0.0004366049598323437),
     (('macb', 'ile'), 0.0004366049598323437),
     (('macb', 'thou'), 0.0004366049598323437),
     (('make', 'vs'), 0.0004366049598323437),
     (('mellon', 'university'), 0.0004366049598323437),
     (('mine', 'eyes'), 0.0004366049598323437),
     (('mine', 'owne'), 0.0004366049598323437),
     (('print', 'statement'), 0.0004366049598323437),
     (('scena', 'secunda'), 0.0004366049598323437),
     (('ten', 'thousand'), 0.0004366049598323437),
     (('three', 'witches'), 0.0004366049598323437),
     (('thy', 'selfe'), 0.0004366049598323437),
     (('worthy', 'thane'), 0.0004366049598323437),
     (('would', 'haue'), 0.0004366049598323437),
     (('among', 'things'), 0.00034928396786587494),
     (('borne', 'woman'), 0.00034928396786587494),
     (('come', 'come'), 0.00034928396786587494),
     (('enter', 'banquo'), 0.00034928396786587494),
     (('enter', 'king'), 0.00034928396786587494),
     (('enter', 'macduffe'), 0.00034928396786587494),
     (('enter', 'rosse'), 0.00034928396786587494),
     (('etext', "shakespeare's"), 0.00034928396786587494),
     (('haile', 'king'), 0.00034928396786587494),
     (('haile', 'macbeth'), 0.00034928396786587494),
     (('hath', 'made'), 0.00034928396786587494),
     (('haue', 'seene'), 0.00034928396786587494),
     (('macb', 'bring'), 0.00034928396786587494),
     (('macbeth', 'macbeth'), 0.00034928396786587494),
     (('malcolme', 'donalbaine'), 0.00034928396786587494),
     (('may', 'see'), 0.00034928396786587494),
     (('old', 'man'), 0.00034928396786587494)]




```python

```
