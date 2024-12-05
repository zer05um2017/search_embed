from enum import Enum


class Products(Enum):
    KEY = "products"


class JsonKeysEnum:
    PRODUCTS = Products


class IndexKey(Enum):
    KEY_SKMALL = "skmall"
    KEY_NSMALL = "nsmall"
    KEY_WMALL = "wmall"
    KEY_SSGDFS = "ssgdfs"
    KEY_SHOPNT = "shopnt"
    KEY_GYMALL = "gymall"


class IndexValue(Enum):
    VALUE_SKMALL = "skmall_index"
    VALUE_NSMALL = "nsmall_index"
    VALUE_WMALL = "wmall_index"
    VALUE_SSGDFS = "ssgdfs_index"
    VALUE_SHOPNT = "shopnt_index"
    VALUE_GYMALL = "gymall_index"


class FaissIndexMapper:
    INDEX_KEY = IndexKey
    INDEX_VALUE = IndexValue

# ============ Reference code ============
# self.index_files = {
#     "open_biz1": "ob_index1",
#     "open_biz2": "ob_index2",
#     "open_biz3": "ob_index3",
#     # "fanrotv_vod"    : "fv_index",
#     "best_prod": "bp_index",
#     "studio_info"    : "si_index",
#     "news_info"      : "ni_index",
#     "e_learning_lv1" : "el_lv1_index",
#     "e_learning_lv2" : "el_lv2_index",
#     "e_learning_lv3" : "el_lv3_index"
# }
