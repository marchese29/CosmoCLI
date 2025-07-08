from typing import Union

JSONPrimitive = Union[int, str, float, None]
JSONObject = dict[str, Union[JSONPrimitive, 'JSONValue']]
JSONList = list['JSONValue']
JSONValue = Union[JSONPrimitive, JSONObject, JSONList]
