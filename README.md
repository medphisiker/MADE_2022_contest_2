# MADE_2022_contest_2
Контест MADE 2022 часть 2.

Полезные советы от участников соревнования набравших много баллов:
1. Есть признак в стиле data leakage и это дата создания файла с картинкой!
TODO:
Надо будет попробовать обучить LAMA с этим признаком.

2. Гораздо лучше работало алгоритмическое решение я почти его сделал, нужно подобрать больше правил и параметров. 
TODO:
Доделать

3. очень хорошо сработал в этой задаче подход кластеризации основанный на плотности (dense based кластеризция).
Действительно все объекты обладают разной формой, получаем все пиксели каждого объекта по его цвету и разделяем их по кластерам.
Надо будет поробовать, возможно кластеризация по графу тоже отлично сработала бы.
TODO:
разобраться в Density-based and Graph-based Clustering.
Материалы:
Density-based and Graph-based Clustering | by Arun Jagota | Towards Data Science
https://towardsdatascience.com/density-based-and-graph-based-clustering-a1f0d45ff5fb

Cluster analysis:. Clustering is a statistical… | by Suresha HP | Nerd For Tech | Jan, 2021 | Medium | MLearning.ai
https://medium.com/mlearning-ai/cluster-analysis-6757d6c6acc9

Обзор алгоритмов кластеризации данных / Хабр
https://habr.com/ru/post/101338/

Поробовать их в деле на данной задаче )
