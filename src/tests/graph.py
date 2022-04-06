import matplotlib.pyplot as plt
import numpy as np

dates = ('12', '13', '14', '15', '16')
y_pos = np.arange(len(dates))
performance = (9, 13, 23, 27, 36)

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, dates, rotation=90)
plt.xlabel('Hospitalized Cases')
plt.title('Cumulative Number of Hospitalized Cases')
plt.tight_layout()
plt.savefig('bar.png')
plt.show()
