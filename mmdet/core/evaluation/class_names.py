import mmcv
import anytree
from anytree.importer import DictImporter
import yaml


def load_tree_from_file(file):
    dct = mmcv.load(file)
    root = DictImporter().import_(dct)
    return root


def wider_face_classes():
    return ['face']


def voc_classes():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def imagenet_det_classes():
    return [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]


def imagenet_vid_classes():
    return [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
        'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
        'watercraft', 'whale', 'zebra'
    ]


def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


def cityscapes_classes():
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def visualgenome_predicates():
    """VG 150 predicates. (50 predicates) """
    return [
        'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between',
        'carrying',
        'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding',
        'in',
        'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of',
        'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to',
        'under',
        'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with'
    ]


def visualgenome_verbs():
    """VG verbs: separated from the VG 150 predicates. (13 verbs)"""
    return [
        'parked', 'standing', 'sitting', 'walking', 'lying', 'laying', 'growing', 'riding', 'flying', 'eating',
        'carrying', 'holding', 'using'
    ]


def visualgenome_prepositions():
    return [
        'above', 'on', 'over',  # pattern "on"
        'under',  # pattern "under"
        'at', 'near', 'and',  # pattern "next/near"
        'behind', 'on back of',  # pattern "behind"
        'in front of',  # pattern "in front of"
        'across', 'against', 'looking at', 'says', 'watching', 'to',  # pattern "opposite"
        'between',  # pattern "between"
        'in',  # pattern "in"
        'attached to', 'belonging to', 'hanging from', 'has', 'of', 'mounted on', 'painted on', 'part of',
        'wearing', 'wear', 'with', 'covered in', 'covering',  # pattern "attached"
        'for', 'from', 'made of', 'playing', 'along'  # others
    ]


def visualgenome_classes():
    """VG 150 classes"""
    return [
        'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird',
        'board',
        'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car',
        'cat',
        'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear',
        'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe',
        'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house',
        'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man',
        'men',
        'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people',
        'person',
        'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing',
        'rock',
        'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink',
        'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie',
        'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase',
        'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra'
    ]


def visualgenome_attributes():
    """VG 200 attributes, collected by TDE (Kaihua Tang et.al., CVPR2020) """
    return [
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'wooden',
        'silver', 'orange', 'grey', 'tall', 'long', 'dark', 'pink', 'standing', 'round', 'tan', 'glass', 'here',
        'wood', 'open', 'purple', 'short', 'plastic', 'parked', 'sitting', 'walking', 'striped', 'brick', 'young',
        'gold', 'old', 'hanging', 'empty', 'on', 'bright', 'concrete', 'cloudy', 'colorful', 'one', 'beige', 'bare',
        'wet', 'light', 'square', 'closed', 'stone', 'shiny', 'thin', 'dirty', 'flying', 'smiling', 'painted',
        'thick', 'part', 'sliced', 'playing', 'tennis', 'calm', 'leather', 'distant', 'rectangular', 'looking',
        'grassy', 'dry', 'cement', 'leafy', 'wearing', 'tiled', "man's", 'baseball', 'cooked', 'pictured', 'curved',
        'decorative', 'dead', 'eating', 'paper', 'paved', 'fluffy', 'lit', 'back', 'framed', 'plaid', 'dirt',
        'watching', 'colored', 'stuffed', 'clean', 'in the picture', 'steel', 'stacked', 'covered', 'full', 'three',
        'street', 'flat', 'baby', 'black and white', 'beautiful', 'ceramic', 'present', 'grazing', 'sandy',
        'golden', 'blurry', 'side', 'chocolate', 'wide', 'growing', 'chrome', 'cut', 'bent', 'train', 'holding',
        'water', 'up', 'arched', 'metallic', 'spotted', 'folded', 'electrical', 'pointy', 'running', 'leafless',
        'electric', 'in background', 'rusty', 'furry', 'traffic', 'ripe', 'behind', 'laying', 'rocky', 'tiny',
        'down', 'fresh', 'floral', 'stainless steel', 'high', 'surfing', 'close', 'off', 'leaning', 'moving',
        'multicolored', "woman's", 'pair', 'huge', 'some', 'background', 'chain link', 'checkered', 'top', 'tree',
        'broken', 'maroon', 'iron', 'worn', 'patterned', 'ski', 'overcast', 'waiting', 'rubber', 'riding', 'skinny',
        'grass', 'porcelain', 'adult', 'wire', 'cloudless', 'curly', 'cardboard', 'jumping', 'tile', 'pointed',
        'blond', 'cream', 'four', 'male', 'smooth', 'hazy', 'computer', 'older', 'pine', 'raised', 'many', 'bald',
        'snow covered', 'skateboarding', 'narrow', 'reflective', 'rear', 'khaki', 'extended', 'roman', 'american'
    ]


def visualgenome_predicate_hierarchy():
    root = load_tree_from_file('data/visualgenome/predicate_hierarchy.yaml')
    predicates = mmcv.load('data/visualgenome/predicate_hierarchy.json')
    return root, predicates


def aithor_classes():
    """aithor 125 classes."""
    return [
        'AlarmClock', 'AluminumFoil', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub',
        'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'BreadSliced',
        'ButterKnife', 'Cabinet', 'Candle', 'CD', 'CellPhone', 'Chair', 'Cloth', 'CoffeeMachine', 'CoffeeTable',
        'CounterTop', 'CreditCard', 'Cup', 'Curtains', 'Desk', 'DeskLamp', 'Desktop', 'DiningTable', 'DishSponge',
        'DogBed', 'Drawer', 'Dresser', 'Dumbbell', 'Egg', 'EggCracked', 'Faucet', 'Floor', 'FloorLamp', 'Footstool',
        'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle',
        'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce', 'LettuceSliced', 'LightSwitch',
        'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan', 'PaperTowel', 'Pen', 'Pencil',
        'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl',
        'RoomDecor', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShelvingUnit', 'ShowerCurtain', 'ShowerDoor',
        'ShowerGlass', 'ShowerHead', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula',
        'Spoon', 'SprayBottle', 'Statue', 'Stool', 'StoveBurner', 'StoveKnob', 'TableTopDecor', 'TargetCircle',
        'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper',
        'ToiletPaperHanger', 'Tomato', 'TomatoSliced', 'Towel', 'TowelHolder', 'TVStand', 'VacuumCleaner', 'Vase',
        'Watch', 'WateringCan', 'Window', 'WineBottle']


def visualgenomegn_classes():
    """generalized visualgenome 3,000 classes"""
    info = mmcv.load('data/visualgenomegn/VGGN-SGG-dicts.json')

    class_to_ind = info['label_to_idx']
    ind_to_classes = list(sorted(class_to_ind, key=lambda k: class_to_ind[k]))

    return ind_to_classes


def visualgenomegn_tokens():
    """generalized visualgenome tokens"""
    info = mmcv.load('data/visualgenomegn/VGGN-SGG-dicts.json')

    token_to_ind = info['token_to_idx']

    ind_to_tokens = list(sorted(token_to_ind, key=lambda k: token_to_ind[k]))

    return ind_to_tokens


dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12', 'VOCDataset'],
    'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
    'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
    'coco': ['coco', 'mscoco', 'ms_coco', 'CocoDataset'],
    'wider_face': ['WIDERFaceDataset', 'wider_face', 'WDIERFace'],
    'cityscapes': ['cityscapes'],
    'visualgenome': ['visualgenome', 'vg', 'VG', 'VG150', 'vg150', 'VisualGenomeDataset', 'VisualGenome'],
    'visaulgenomekr': ['visualgenomekr', 'vgkr', 'VGKR', 'vg200', 'VG200', 'VisualgenomeKRDataset', 'VisualGenomeKR'],
    'visualgenomegn': ['visualgenomegn', 'vggn', 'VGGN', 'VisualGenomeGN', 'VisualgenomeGNDataset'],
    'aithor': ['aithor', 'AithorDataset', 'ai2thor', 'AITHOR']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def get_predicates(dataset):
    """Get predicate names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_predicates()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def get_attributes(dataset):
    """Get attributes names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_attributes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def get_verbs(dataset):
    """Get verb names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_verbs()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def get_prepositions(dataset):
    """Get pure predicate names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_prepositions()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def get_predicate_hierarchy(dataset):
    """Get predicate names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_predicate_hierarchy()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def get_tokens(dataset):
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_tokens()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels
