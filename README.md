# Deep Learning 
**Auteurs**\
Thomas Danguilhen\
Emil Răducanu\
Céline Goncalves\
Arthur Sarmini Det Satouf

# Exécution du projet 
- créer un notebook 
- insérer les lignes de codes ci-dessous

```python
import ai
model=ai.ai_plantes(BATCH_SIZE=32)
model.preporcess(isMacOs=True,out_put="sigmoid",min_sample=False )
model.create_model(balance=True)
his=model.fit(epochs=50)

model.print_multilabel_confusion_matrix()
model.get_roc_curve()
```
