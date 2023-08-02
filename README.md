# 실습에 나온 Python 관련 정리

# 1. matplotlib

## .plot()

- .plot에 X축, Y축 데이터 넣고, 선과 마커를 정할 수 있다.
    - ‘k.’이 의미하는 것은 검은색(blac**k**)과 점 모양이다.
    - [https://matplotlib.org/cheatsheets/_images/cheatsheets-1.png](https://matplotlib.org/cheatsheets/_images/cheatsheets-1.png)에서 Marker, Colors, Lines에 무엇을 쓸 수 있는지 나와 있으므로 그래프 그릴 때 참고
        
        ```python
        # ML 실습 Chapter 2.ipynb
        plt.figure()
        plt.plot(x_ideal, y_ideal, label='distribution')
        plt.plot(x, y, 'k.', label='sampled with noise') # 검은색 점들로 그린다.
        plt.legend()
        ```
        
        ```python
        # ML 실습 Chapter 2.ipynb
        plt.plot(x, y_pred, 'k-') # 검은색이면서 라인을 그린다.
        ```
        

# 2. numpy

## np.vander(데이터 X, 차수)

- [https://numpy.org/doc/stable/reference/generated/numpy.vander.html](https://numpy.org/doc/stable/reference/generated/numpy.vander.html)
- 데이터에 들어간 값들을 이용하여 다항식 행렬을 만들어 준다.

$$
np.vander([X_1, X_2], n) = 
\begin{bmatrix} 
    X_1^n & X_1^{n-1} & … & X_1^2 & X_1^1 & 1\\ 
    X_2^n & X_2^{n-1} & … & X_2^2 & X_2^1 & 1 
\end{bmatrix}
$$

```python
# ML 실습 Chapter 2.ipynb
model.fit(**np.vander(x, degree + 1)**, y)
```

---

## np.enumerate(배열)

- 배열에서 인덱스 번호와 값을 반복해서 꺼내 올 수 있다. 주로 for ~ in 과 같이 쓰임

## np.ndenumerate(다차원 배열)

- np.enumerate와 같은 역할을 하는데, 여러 차원의 배열에서 가져올 때 쓴다.
- 다차원 배열이므로, 인덱스는 튜플로 리턴됨. 아래 예시에서 i, j는 2차원 배열의 row, column을 의미함.
- v는 배열에 있는 값

```python
# ML 실습 Chapter 3.ipynb
Z = np.zeros((B0.size, B1.size))

# Calculate Z-values (RSS) based on grid of coefficients
for (i, j), v in np.ndenumerate(Z):
    Z[i,j] = np.sum((Y - (xx[i,j] + X.reshape(-1) * yy[i,j])) ** 2)
```

```python
# 간단한 예제
>>> Z = np.zeros((3, 4))
>>> for (i, j), v in np.ndenumerate(Z):
...     print("index:", i, j, ", value:", v)
...
index: 0 0 , value: 0.0
index: 0 1 , value: 0.0
index: 0 2 , value: 0.0
index: 0 3 , value: 0.0
index: 1 0 , value: 0.0
index: 1 1 , value: 0.0
index: 1 2 , value: 0.0
index: 1 3 , value: 0.0
index: 2 0 , value: 0.0
index: 2 1 , value: 0.0
index: 2 2 , value: 0.0
index: 2 3 , value: 0.0
```

---

# 3. Pandas

## df[[컬럼명]]

- .fit() 함수의 X_train은 항상 2차원 배열이어야 하기 때문에?

```python
# ML 실습 Chapter 4.ipynb
X_train = **df[['balance']] # ['balance']가 아니라 [['balance']]**
y_train = df['default']

clf = LogisticRegression(solver='newton-cg')
clf.fit(X_train, y_train)
```

# 3. 내장 함수

## zip(*args, **kwargs)

- 인자로 입력되는 것들을 하나씩 꺼내서 조합한다.
- 아래 코드에서 labels와 handles에서 하나씩 꺼내서 튜플로 만든다.

```python
# ML 실습 Chapter 2.ipynb
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
```

```python
# zip에 대한 간단한 예
numbers = (1, 2, 3)
letters = ("A", "B", "C")
pairs = list(zip(numbers, letters)) # 결과: [(1, 'A'), (2, 'B'), (3, 'C')]
```

# 4. Python 문법

## 문자열 앞의 f

- Python 3.6부터 사용 가능. 변수를 바로 출력할 수 있다.

```python
a = 1
b = 2
text = **f**"{a}는 {b}보다 작다." # 출력: 1는 2보다 작다.
```

---

## 문자열 앞의 r

- String 형태를 Raw String으로 변경. 문자 그대로 출력됨
- 아래 예시에서는 scatter3D 그래프 그릴 때 label에 Latex 표현으로 $\beta_0$ 같은 것을 적기 위해 사용
- 정규 표현식(Regex)이나 Latex 표현식을 사용할 때 주로 쓰는 듯

```python
# ML 실습 Chapter 3.ipynb
# Minimized RSS
**min_RSS** = r'$\beta_0$, $\beta_1$ for minimized RSS'
min_rss = np.sum((regr.intercept_ + regr.coef_ * X - Y.values.reshape(-1,1)) ** 2)
...
...
ax2.scatter3D(regr.intercept_, regr.coef_[0], min_rss, c='r', label=**min_RSS**)
```

![20230802_01](https://github.com/reiple/ds2_python/assets/6015403/81fd229e-c46f-420e-b3af-d995768edfb4)

---

## 변수명 대신 언더바(_)

- 함수에서 값을 받는 부분에 변수명을 써야 하는데 무시하고 싶을 때 사용

```python
# ML 실습 Chapter 4.ipynb
# factorize() returns two objects: a label array and an array with the unique values
df['default'], target_names = df['default'].factorize()
df['student'], _ = df['student'].factorize()
```

- 긴 숫자의 자리 수 구분을 위해서 사용하기도 함. 0 몇 개를 적었는지 알기 편하다.

```python
a = 100000000 # 1억
b = 100_000_000 # 자리수 구분을 위해 _ 사용. 1억
```
