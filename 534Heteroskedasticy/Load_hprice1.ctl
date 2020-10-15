OPTIONS(errors=2,load=500,skip=1)

LOAD DATA 
INFILE 'C:\Users\jfras\OneDrive\UND\534AppliedEcon\Datasets\hprice1.csv' 
TRUNCATE 
INTO TABLE "C##HET"."HPRICE1" 
fields terminated by "," 
TRAILING NULLCOLS 
(
    PRICE,ASSESS,BDRMS,LOTSIZE,
	SQRFT,COLONIAL,LPRICE,LASSESS,LLOTSIZE,
	LSQRFT 
)
