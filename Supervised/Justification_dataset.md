# Justification du Dataset (Supervised Learning)

## Contexte

Pour le projet CVRIE, il nous avait été demandé de passer par deux types d'entraînements, chacun à l'opposé de l'autre. En effet, ces deux types d'entraînement étaient :
- Le <u>"Supervised Learning"</u> (Apprentissage supervisé), utilisant un dataset labelisé.  
- Le <u>"Unsupervised Learning"</u> (Apprentissage non-supervisé), utilisant un dataset non labelisé.

Dans la première partie, l'apprentissage supervisé, il nous a fallu trouver un dataset d'images avec seulement leurs labels. La seule obligation de ce dataset était qu'il devait être un "medical" imaging dataset. Nous avons donc choisi un dataset porté sur les tumeurs cérébrales.

## Quelque données sur les tumeurs cérébrales

Une tumeur au cerveau est une masse de cellules anormales qui se multiplient dans le tissu cérébral ou à proximité de celui-ci. Elle cause des maux de tête persistants, des crises d'épilepsie, des troubles de la vision, de l'équilibre ou de la parole ou encore des faiblesses musculaire ou fatigue inexpliquée. Et bien sûr, la mort.

Les tumeurs cérébrales sont l'une des causes de mortalité les plus importantes dans le monde, c'est d'ailleurs la première cause de mortalité par cancer chez les enfants et les jeunes adultes.
On comptabilise environ 321 400 nouveaux cas, et près de 248 300 décès par an, dont 6 000 nouveaux cas et 4000 décès en France !

Ce n'est donc pas un sujet à prendre à la légère et surtout, c'est un sujet dans lequel les recherches doivent évoluer pour en faire reculer la mortalité.

## Pourquoi avoir choisi ce dataset ?

Comme dit précédemment, toute recherche sur ce sujet est utile et importante. Des milliers de personnes souffrent et meurent de ce cancer, il est donc dans notre devoir à tous de trouver un moyen de stopper ce fléau, ou du moins de le ralentir.

Des raisons privées nous ont également poussé à nous pencher sur ce sujet. 

Mais aussi car l'un d'entre nous souhaiterait travailler dans ce sujet précis à l'avenir. La recherche médicale aidée par l'IA, pour la lutte contre le cancer, est un des projets professionnels dans lequel l'un de nos membres souhaiterait évoluer.

Ce dataset contient 4 types d'images. 
- Des images de cerveaux sains.
- Des images de cerveaux touchés par une tumeur nommée : Glioma
- Des images de cerveaux touchés par une tumeur nommée : Meningioma
- Des images de cerveaux touchés par une tumeur nommée : Pituitary tumor

Elles permettent de bien comparer les différents types de tumeurs et de savoir si le patient est en danger imminent ou non.

- Les images des cerveaux sains permettent, et cela tombe sous le sens, de savoir si le patient n'a pas de tumeurs, et donc qu'il n'a pas besoin de recevoir des traitements contre le cancer.

- Les images avec Glioma (Gliome en français), sont LES PLUS importantes à analyser. En effet, entraîner l'IA à reconnaître ce type de tumeurs, qui est l'une des plus dangereuses des tumeurs cérébrales, permettrait de les diagnostiquer bien plus vite. Les Gliomes sont l'ensemble des tumeurs cérébrales issues du "tissu de soutien neuronal" ou "glie". Ils sont bien souvent malins, c'est-à-dire "cancéreux et rapides" (Grades 3/4 de la tumeur), signifiant qu'en à peine 6 mois ou jusqu'à 4 ans au maximum, elle sera fatale pour l'individu. La forme la plus commune de gliome est "le glioblastome", elle est de grade 4, signifiant la mort dans les 6 mois à 2 ans de l'individu concerné. Il est donc réellement nécessaire d'apprendre à diagnostiquer celui-ci le plus rapidement possible.

- Les images avec Meningioma (un méningiome) sont très importantes à analyser, car c'est la tumeur primitive (première tumeur, point de départ d'un potentiel cancer) la plus fréquente du cerveau. Bien que 80 à 90% des cas soient bénins et soient tirés d'affaire 5 ans après être diagnostiqués, elle est la cause de 8 875 décès aux États-Unis. Comme son nom l'indique, elle se forme au niveau des méninges, se trouvant entre l'os du crâne et le cerveau.

- Et enfin, les images avec Pituitary tumor (tumeur pituitaire ou tumeur hypophysaire) se trouvent à la base du cerveau, au niveau de la glande pituitaire (l'hypophyse). Elle est donc facilement identifiable. Ce type de tumeurs est majoritairement non cancérigène, elle n'en reste pas moins dangereuse mais potentiellement moins urgente à traiter que le Gliome.

Ce dataset est donc parfait pour entraîner une IA sur cette problématique. En plus de permettre de détecter les tumeurs, notre IA sera capable de détecter le type de la tumeur, permettant de rediriger les patients au bon endroit et vers les bons traitements.