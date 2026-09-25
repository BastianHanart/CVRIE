# Justification du Dataset (Supervised Learning)

## Contexte

Pour le projet CVRIE, il nous avait été demandé de passer par deux types d'entraînements, chacun à l'opposé de l'autre. En effet, ces deux types d'entraînement étaient :
- Le <u>"Supervised Learning"</u>, utilisant un dataset labelisé.  (Partie 1)
- Le <u>"Unsupervised Learning"</u>, utilisant un dataset non labelisé. (Partie 2)

Dans la première partie, l'apprentissage supervisé, il nous a fallu trouver un dataset d'images avec seulement leurs labels. La seule obligation de ce dataset était qu'il devait être un "medical" imaging dataset. Nous avons donc choisi un dataset porté sur les tumeurs cérébrales.

## Pourquoi avoir choisi ce dataset ?

Ce dataset contient 4 types d'images. 
- Des images de cerveaux sains.
- Des images de cerveaux touchés par une tumeur nommée : Glioma
- Des images de cerveaux touchés par une tumeur nommée : Meningioma
- Des images de cerveaux touchés par une tumeur nommée : Pituitary

Elles permettent de bien comparer les différents types de tumeurs et de savoir si le patient est en danger imminent ou non.

- Les images des cerveaux sains permettent, et cela tombe sous le sens, de savoir si le patient n'a pas de tumeurs, et donc qu'il n'a pas besoin de recevoir des traitements contre le cancer.

- Les images avec Glioma, sont les plus importantes à analyser. En effet, entraîner l'IA à reconnaître ce type de tumeurs, qui est l'une des plus dangereuses des tumeurs cérébrales, permettrait de les diagnostiquer bien plus vite. Les Gliomes sont l'ensemble des tumeurs cérébrales issues du "tissu de soutien neuronal" ou "glie". Ils sont bien souvent malins, c'est-à-dire "cancéreux et rapides", signifiant qu'en à peine 6 mois ou jusqu'à 4 ans au maximum, elle sera fatale pour l'individu. La forme la plus commune de gliome est "le glioblastome", elle est de grade 4, signifiant la mort dans les 6 mois à 2 ans de l'individu concerné. Il est donc réellement nécessaire d'apprendre à diagnostiquer celui-ci le plus rapidement possible.

- Les images avec Meningioma sont très importantes à analyser, car c'est la tumeur primitive (première tumeur, point de départ d'un potentiel cancer) la plus fréquente du cerveau. Bien que 80 à 90% des cas soient bénins et soient tirés d'affaire 5 ans après être diagnostiqués, elle est la cause de 8 875 décès aux États-Unis. Comme son nom l'indique, elle se forme au niveau des méninges, se trouvant entre l'os du crâne et le cerveau.

- Et enfin, les images avec Pituitary tumor (tumeur pituitaire ou tumeur hypophysaire) se trouvent à la base du cerveau, au niveau de la glande pituitaire (l'hypophyse). Elle est donc facilement identifiable. Ce type de tumeurs est majoritairement non cancérigène, elle n'en reste pas moins dangereuse mais potentiellement moins urgente à traiter que le Gliome.

Ce dataset est donc parfait pour entraîner une IA sur cette problématique. En plus de permettre de détecter les tumeurs, notre IA sera capable de détecter le type de la tumeur, permettant de rediriger les patients au bon endroit et vers les bons traitements.
