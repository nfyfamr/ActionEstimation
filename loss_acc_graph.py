import matplotlib.pyplot as plt

log_fn = './models/training.log'


history = {'loss': [], 'acc': []}
with open(log_fn, "r") as f:
    for line in f:
        line = line.split(',')
        loss = line[-2].split(':')[1].strip()
        loss = float(loss)
        acc = line[-1].split(':')[1].strip()
        acc = float(acc)

        history['loss'].append(loss)
        history['acc'].append(acc)


step_size = len(history['loss'])

plt.subplot(2,1,1)
plt.plot(range(step_size), history['loss'], label='Loss', color='red')
plt.title('Loss history')
plt.xlabel('step')
plt.ylabel('loss')

plt.subplot(2,1,2)
plt.plot(range(step_size), history['acc'], label='Accuracy', color='blue')
plt.title('Accuracy history')
plt.xlabel('step')
plt.ylabel('accuracy')

plt.savefig('./models/result')