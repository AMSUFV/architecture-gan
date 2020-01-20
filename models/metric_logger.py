import datetime
import tensorflow as tf


class MetricLogger:
    """
    Para mantener el principio de responsabilidad única se implementa un logger de métricas, el cual se encargará de
    recoger y escribir las métricas seleccionadas en los logs.
    Para emplear esta clase, crear un nuevo MetricLogger en el init, elegir  y las métricas que se quieren emplear a
    través de add_metric
    """
    def __init__(self, log_dir='logs', default_metrics=False, datasets=None):
        """
        Se inicializa indicándole en qué directorio se guardarán los logs. Por defecto se guardarán en el directorio
        logs.

        :param log_dir: String. Ruta a donde se guardarán los logs.
        """

        # Un diccionario para entrenamiento y otro para validación
        if datasets is None:
            datasets = ['train', 'validation']

        if default_metrics:
            self.writers = {'train': self.set_logdirs(log_dir, 'train'),
                            'validation': self.set_logdirs(log_dir, 'validation')}
            self.metrics = {'train': dict(), 'validation': dict()}
            self.metrics['train']['loss'] = tf.keras.metrics.Mean(dtype=tf.float32)
            self.metrics['validation']['loss'] = tf.keras.metrics.Mean(dtype=tf.float32)
        else:
            self.writers = dict()
            self.metrics = dict()
            for dataset in datasets:
                self.writers[dataset] = self.set_logdirs(log_dir, dataset)
                self.metrics[dataset] = dict()

    @staticmethod
    def set_logdirs(path: str, name: str):
        """
        Establece dónde se guardarán los logs del entrenamiento

        :param path: String. Path a la carpeta donde se guardarán los logs del entrenamiento
        :param name: String. Nombre de la subcarpeta.
        :return: file writer, file writer
        """
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = path + r'\\' + current_time + r'\\' + name
        # val_log_dir = path + r'\\' + current_time + r'\val'

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        return train_summary_writer # val_summary_writer

    def function(self, func, dataset):
        def wrapper(*args, **kwargs):
            values = func(*args, **kwargs)
            self.metrics[dataset] = values
        return wrapper


    def add_metric(self, dataset: str, tag: str, value):
        """
        Método para añadir métricas escalares. Ej:
        metric_logger.add_metric(dataset='train', tag='loss', value=tf.keras.metrics.Mean())

        :param dataset: String; train o validation. Dataset al que pertenecen las métricas
        :param tag: String. Tag de la métrica; en Tensorboard se agruparán en función del tag.
        :param value: tf.keras.metrics.*. Métrica escalar a loggear (Mean, BinaryAccuracy, etc).
        """

        self.metrics[dataset][tag] = value

    def update_metric(self, dataset: str, tag: str, new_value):
        """
        Método para actualizar las métricas. Si la métrica no está en las recogidas se le asignará un valor en función
        de la etiqueta.

        :param dataset: String; train o validation.
        :param tag: String. Tag de la métrica; en Tensorboard se agruparán en función del tag.
        :param new_value: Valor actualizado para la métrica.
        :return:
        """
        if tag not in self.metrics[dataset].keys():
            if 'loss' in tag:
                self.metrics[dataset][tag] = tf.keras.metrics.Mean()
            elif 'acc' in tag:
                self.metrics[dataset][tag] = tf.keras.metrics.BinaryAccuracy()

        self.metrics[dataset][tag](new_value)

    def write_metrics(self, dataset: str, epoch: int):
        """
        Método para escribir en archivo las métricas seguidas.

        :param dataset: String; train o validation. Dataset al que pertenecen las métricas
        :param epoch: int. Step en la gráfica; correspondiente con la época
        """

        with self.writers[dataset].as_default():
            for tag in self.metrics[dataset].keys():
                tf.summary.scalar(tag, self.metrics[dataset][tag].result(), step=epoch)

    def write_metric(self, dataset: str, tag: str, value, epoch: int, metric_type='image', **kwargs):
        """
        Método para escribir métricas no incluidas en los diccionarios de train o validation por no ser escalares o por
        tener una casúistica diferente.
        Para escribir audio es necesario especificar el valor 'sample_rate' al llamar a la función

        :param dataset: String; train o validation. Dataset al que pertenecen las métricas
        :param value: tf image stack
        :param tag: String. Tag de la métrica; en Tensorboard se agruparán en función del tag.
        :param epoch: int. Step en la gráfica; correspondiente con la época
        :param metric_type: str. Tipo de métrica.
        """

        with self.writers[dataset].as_default():
            if metric_type == 'image':
                tf.summary.image(tag, value, step=epoch)
            elif metric_type == 'audio':
                if 'sample_rate' not in kwargs.keys():
                    raise Exception('sample_rate not provided')
                tf.summary.audio(tag, value, sample_rate=kwargs['sample_rate'], step=epoch)
            else:
                raise Exception('metric type not supported')

    def reset_metrics(self, dataset):
        for key in self.metrics[dataset].keys():
            # value.reset_states()
            self.metrics[dataset][key].reset_states()
