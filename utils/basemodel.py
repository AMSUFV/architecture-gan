class BaseModel:
    """Clase base de la que el resto de modelos heredan

    Esta clase pretende forzar la implementación de unos métodos comunes entre modelos (configuración de pesos,
    generación del dataset, entrenamiento, validación, predicción, etc.), obteniendo así coherencia y similitud
    de uso entre los mismos.
    """

    def set_weights(self, **kwargs):
        """Método para establecer los pesos de las redes empleadas

        Este se emplea de manera automática durante la inicialización de los modelos si no se
        provee una ruta a los pesos. Puede emplearse para utilizar pesos obtenidos en anteriores
        ejecuciones.
        """
        raise NotImplementedError

    @staticmethod
    def save_weights(model, name):
        """Método para guardar los pesos de los modelos

        Se empleará al inicializar los modelos si no se han provisto rutas a pesos iniciales; los
        nuevos pesos se guardarán para futuras ejecuciones.
        """
        raise NotImplementedError

    @staticmethod
    def get_dataset(**kwargs):
        """Método propio de cada modelo para generar su dataset

        Los parámetros de este método variaran entre diferentes modelos
        """
        raise NotImplementedError

    def fit(self, train_ds, test_ds, epochs, **kwargs):
        """Método de entrenamiento.

        Se entrenará y validará el modelo con un datasets de entrenamiento y validación durante un número determinado
        de épocas.
        """
        raise NotImplementedError

    @staticmethod
    def validate(**kwargs):
        """Método de validación.
        """
        raise NotImplementedError

    def predict(self, **kwargs):
        """Método in-out; uso del modelo como tal

        Dado un input se obtendrá un output que podrá almacenarse
        """
        raise NotImplementedError
