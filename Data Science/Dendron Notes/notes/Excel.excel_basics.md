---
id: xqna7dmp440xykq4yvy4dk1
title: Excel Basics
desc: ''
updated: 1688743573186
created: 1688534759469
---

1. Getting random data using `=RAND()` or `=RANDBETWEEN()`

### Mean, Median, Mode

```excel
<!-- Mean -->
=AVERAGE()

<!-- Median -->
=MEDIAN()

<!-- Mode -->
=MODE()
=MODE.SNGL()
=MODE.MULT()

```

![Mean Median Mode](<assets/images/Excel/Mean Median Mode.png>)

### Minimum, Maximum, Quartile

```excel
=MIN(B21:B50)
=MAX(B21:B50)
=QUARTILE(B21:B50,1)
=QUARTILE(B21:B50,2)
=QUARTILE(B21:B50,3)
```

![Min Max Quartile](<assets/images/Excel/Min Max Quartile.png>)

### Variance and Standard Deviation

```excel
<!-- To calculate variance -->

=VAR.S(B55:B74)

<!-- To calculate Standard Deviation -->

=STDEV.S(B55:B74)
```

![Variance and Standard Deviation](<assets/images/Excel/Variance Std Deviation.png>)

![Alt text](<assets/images/Excel/Variance and Std Deviation Formula.png>)

### Introduction to central limit theorem

![Normal Distribution Curve](<assets/images/Excel/Normal Distribution Curve.png>)

![Normal Distribution with standard deviations](<assets/images/Excel/Normal Distribution with Standard Deviation.png>)

#### Intorduction to Margin of Errors

Standard error = **sigma**/sqrt(N)

Margin of error is standard error times z-score

Z-score is the number of standard deviation from mean

A z-score of 1 includes about 68% of values

![Commonly Used Z-score](<assets/images/Excel/Commonly Used Z-scores.png>)

Calculating Sample margin of error

![Sample Margin of Error](<assets/images/Excel/Sample Margin of Error.png>)

### _Sources of Error_

1. Usion non-random samples
2. Investigator bias
3. Working with old data
4. Basing policy on a survey on experiment with a small sample

## _**Group Data using Histogram**_

## To generate a normal distribution in Excel, you can use the following formula

```excel
<!-- Generate random number using normal distribution -->

=NORMINV(RAND(), MEAN, STANDARD_DEVIATION)

```

### Calculating running SUM, and Average

![Running Sum, and Average](<assets/images/Excel/Running Sum, and Average.png>)