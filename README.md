# Proyecto AP Portfolio: Pablo Martín Berna Y Mauro García Lorenzo
(La IA Generativa se ha utilizado en muchas fases del proyecto; tanto en la fase de planteamiento del problema, como en la implementación y refinado de los resultados).
El objetivo de este proyecto era el de entrenar un modelo que fuera capaz de predecir cuál es el planificador apropiado (de los entrenados), para un problema a tratar.
Para ello cogeremos 1000 problemas del repositorio de la competición IPC utilizado para la práctica de Fast Downward (directorio _{ROOT_DIR}/data/}_).

De estos 1000 problemas obtendremos muchas características utilizando los scripts que se encuentran en el directorio _{ROOT_DIR}/feature_extraction/_. La tabla con las características de cada problema se encuentra en {ROOT_DIR}/all_features.csv.
El número de características que obtuvimos para cada problema era muy alto, por lo que iteramos bastante a la hora de entrenar el modelo para encontrar las características más determinantes. 
La selección de características y entrenamiento el modelo se realiza mediente el script {ROOT_DIR}/post_process/train_selector.py. 

Una vez que tenemos las características, utilizamos el script _{ROOT_DIR}/problem_solving.py_. Los logs de las ejecuciones no están subidos al repositorio, ya que ocupan mucho espacio.

Para analizar los logs y obtener estadísticas de las ejecuciones utilizamos dos scripts: _{ROOT_DIR}/post_process/analyze_results.py_, y _{ROOT_DIR}/post_process/parse_fg_logs_to_csv.py_. 
Estas estadísticas se recogen en _{ROOT_DIR}/post_process/results_analysis_. 

Por último, hacemos un postproceso con {ROOT_DIR}/analyze_distributions, script que nos permitía obtener conclusiones de 
las características que tiene en cuenta el modelo para seleccionar un planificador u otro, y extraer las conclusiones finales del proyecto.

