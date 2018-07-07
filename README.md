# detectarDados

Se pretende detectar, localizar y clasificar dados en imágenes de  dados tirados en una mesa. El objetivo final es extraer el puntaje de cada dado. Son dados d10.

Etapas del proyecto


* [ ] generar un dataset anotado
    + [x] sacar fotos
    + [x] cortarlas
    + [ ] hacer una gui para marcar bounding boxes y puntajes
    + [ ] generar las anotaciones (bounding boxes y puntajes)
- [ ] Testear ideas:
    + [ ] segmentarlas y extraer blobs binarizados
    + [ ] usar OCR en los blobs
    + [ ] uasr SURF para extraer los 64 features y pasarlos por una red sencilla 
