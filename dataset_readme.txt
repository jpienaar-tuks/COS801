DATASETS LICENSE
================ 
This work is licensed under a Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.

NOTICE
======
This directory contains the Contract Elements datasets and dictionaries,
as described in the paper:

Ilias Chalkidis, Ion Androutsopoulos, and Achilleas Michos, “Extracting 
Contract Elements”. Proceedings of the 16th International Conference on 
Artificial Intelligence and Law (ICAIL’17), London, UK, 2017.

Please ensure you have carefully read both the paper and this document, 
and also examined several sample files before proceeding to further 
experimentation.                           


There are two main directories. 
- The first directory (/datasets) contains the contracts in encoded 
format for privacy issues. 
- The second directory (/dictionaries) contains multiple CSV files that 
provide the features for the encoded vocabularies. Each CSV file has been 
written with a new line (\n) delimiter between different lines (tokens) 
and a semicolon (;) separator between different columns (token_id, 
features[1:n]) of each line. We strongly recommend to open some 
contract files with a text editor and the dictionaries using a CSV file 
reader in order to get familiarised with the encoded format, which is firmly 
explained in the rest of this document.

These are the following directories (datasets):

* Unlabelled Contracts Dataset: 
================================
- The directory (datasets/unlabelled_contracts) contains multiple
compressed files (unlabelled_contracts_X.tar.gz), each one with a different 
set of contracts. These sets do not correspond into different contract types
or any other classification.The split of the corpus is just for practical 
reasons (e.g. avoid single point of failure during the download process 
or maximum file size limitations in specific filesystems).
- These compressed files contain contracts, that have not been annotated 
and can only been used for unsupervised tasks, like train word embeddings 
(e.g. word2vec, glove, etc.) or extract probabilistic features over
the respective vocabulary.
- Each contract contains a number of sentences, already split between 
different lines (one sentence per line) using the default NLTK Sentence 
Tokenizer. The sentences have been also tokenized (split) in terms of 
words using the default NLTK word tokenizer (TreebankWordTokenizer). Tokens
are split with a single space. All digits have been pre-processed and replaced 
with a single character, so  numbers having the same format have been encoded 
in the same manner (e.g. '12', '13' both have the same code, '12.3', '24.7' 
have another single code). All words that are out of vocabulary have been
replaced with the meta-character 'UNK'.

For example, a sample file in this corpus contains text such as:

2 13 455 43 897 87789 3464 564333 3462 245 67868 85278 34535 6 45 37 78976 
UNK 34635 45456 568 568 678 798 UNK 67 456 2345 9368 235 98545 346 87365

- There is a related CSV file (dictionaries/word_embeddings.csv), which
contains the pre-trained embeddings that were used during experiments, as 
also mentioned in the paper. Each line of the .csv file contains a pair: 
the code of the specific token (e.g. 3464)  in the first column and 200 
real number values of its word embedding in the rest of the columns.


* Labelled Contracts Datasets:
==============================
- The directory (datasets/labelled_contracts) contains two main subdirectories:
(i)  the first one (elements_contracts) contains contracts annotated for the 
following contract elements:  Contract Title, Contract Parties, Start Date, 
Effective Date, Termination Date,  Contract Period, Contract Value, Governing Law, 
Jurisdiction, Legislation Refs. 
(ii) the second one (clauses_contracts) contains contracts annotated for Clause 
Headings.

- Both directories have two subdirectories:
(i)  The training directory (/train), which contains the contracts used for 
training and validation purposes.
(ii) The test directory (/test), which contains the contracts used for 
testing purposes.

Both of these subdirectories contain contracts in the same encoded format, 
which is different from the unlabelled dataset. The contracts (.txt files) 
have been pre-processed also using NLTK sentence and word tokenizers. In 
contrast with the unlabelled dataset, the sentences are not split between
different lines. Line-breaks inside the text are real line-breaks used in 
the initial contracts. Hence, we strongly recommend everyone to use line-breaks 
accordingly. During our own experimentation, we represented line-break as a 
specific token and in each case we assign zeros in its feature representation. 
Each line has been split to tokens,  which have been encoded in the following 
format TOKEN_XXX[CATEGORY]. XXX refers to a specific token according to the 
represented word. In this encoded format, the encoding of the CATEGORY field
is the following:

CONTRACT ELEMENT	   CATEGORY ENCODING
------------------------ + ------------------
None			—> 0
Contract Title 		—> TIT
Contract Party		—> CNP
Start Date		—> STD
Effective Date		—> EFD
Termination Date	—> TED
Contract Period		—> PER
Contract Value		—> VAL
Governing Law		—> GOV
Jurisdiction		—> JUR
Legislation Refs.	—> LEG
Clause Headings		—> HEAD


For example, a sample file in the annotated corpus contains text like the
following part:

TOKEN_2888[0] TOKEN_2889[0] 
TOKEN_1490[TIT] TOKEN_6[TIT] 
TOKEN_15[0] TOKEN_6[0] TOKEN_2384[0] TOKEN_263[0] TOKEN_28816[STD] TOKEN_28[STD] 
TOKEN_25[STD] TOKEN_4376[STD] TOKEN_19[STD] TOKEN_1530[STD] TOKEN_31[0] TOKEN_78167[CNP] 
TOKEN_5565[CNP] TOKEN_1539[CNP] TOKEN_19[0] TOKEN_66[0] TOKEN_12279[0] TOKEN_2207[0] 
TOKEN_2167[0] TOKEN_22[0] TOKEN_2191[0] TOKEN_408[0] TOKEN_26[0] TOKEN_413[0] 
TOKEN_1660[0] TOKEN_8133[0] TOKEN_194[0] TOKEN_57[0] TOKEN_10943[0] TOKEN_74489[0] 
TOKEN_61[0] TOKEN_25[0] TOKEN_26[0] TOKEN_8418[0] TOKEN_8419[0] TOKEN_7949[0] 
TOKEN_22[0] TOKEN_928[0] TOKEN_78[0] TOKEN_17205[0] TOKEN_562[0] TOKEN_208[0] 
TOKEN_508[0] TOKEN_33534[0] TOKEN_25[0] TOKEN_62450[0] TOKEN_50[0] TOKEN_73[0]

— There is a related CSV file (dictionaries/encoded_vocabulary.csv), which 
contains for each encoded token (TOKEN_CODE): the code of this specific word 
(WORD_EMBEDDING_CODE) that matches with the word embeddings file in order to 
find the related word embedding, its POS tag (POS_TAG) and the hand-crafted 
features. The hand-crafted features are organized in the next columns labeled 
with the related names: GENERAL_[1-14], TITLE_[1-3], PARTIES_[1-7], SDATE_[1-5], 
EFDATE_[1-5], TDATE_[1-4], GOV_LAW_[1-4], JUR_[1-4], LEG_[1-5], VALUE_[1-4],
PERIOD_[1-5], HEAD_[1-6]

— There is also a related CSV file (dictionaries/pos_tag_embeddings.csv), which 
contains the POS tag embeddings.

— During training we strongly recommend the use of pseudo-zones as described in 
the paper. The test contracts of the labeled dataset include gold (correct) 
annotations of the extraction zones per contract element type in an XML format 
(e.g. <ZONE_NAMING> … </ZONE_NAMING>). The annotation names of these zones 
are the following and are always coming in pairs (start-end):

ZONE_NAMING	   CONTRACT ELEMENTS
---------------- + --------------------------------------------------------------- 
COVER_PAGE  	-> Contract Title, Contract Parties, Start Date and Effective Date
INTRODUCTION	—> Contract Title, Contract Parties, Start Date and Effective Date
TERMINATION	—> Termination Date
TERM		—> Contract Period, Termination Date
VALUE		—> Contract Value
GOVERNING_LAW	—> Governing Law
JURSDICTION	—> Jurisdiction
MISCELLANEOUS	—> Governing Law, Jurisdiction
MAIN_BODY*	—> Clause Headings
(Full Text)**	-> Legislation Refs.

* This annotation type is only included in the clause headings test dataset. 
The rest of them are only included in in the elements test dataset. 
For experimental testing and comparisons, you will have to create pseudo-zones 
of 20 words before/after each line-break (\n) character across the main body.

** Legislation refs. have to be tested on the elements test dataset.Specific 
zones are not provided since the legislation references can be found all over
the contracts. For experimental testing and comparisons, you will have to 
create pseudo-zones of 20 words before/after the following words: Act, ACT, 
Code, CODE, Regulation, REGULATION, Amendment, AMENDMENT, Treaty, TREATY. 
These words cover the 99,9% of the legislation refs mentioned in the annotated 
contracts.

The encoded tokens of these words (e.g. Act, ACT, etc.) are: 194, 264, 277, 
1489, 2291, 4110, 5559, 6052, 7133, 11423, 12963, 18111, 23005, 23387, 34425, 
53519, 54580, 56472, 57386, 76355, 103593, 107366, 165333, 165336. There are 
multiple encoded tokens per word since each word could be coupled with one or
more POS tags, which brings the final number of the encoded tokens from 10 to 
24.
 

CONTACT
=======
For any further issues, please contact: ihalk@aueb.gr

The paper is available from:
<https:// http://nlp.cs.aueb.gr/publications.html> and 
<http://www.aueb.gr/users/ion/publications.html>.

I. Chalkidis, I. Androutsopoulos and A. Michos 

This file was last updated: May 18, 2017.

Ilias.Chalkidis@di.ku.dk
