import os
import sys
import json


def calculate_sum(a,b):
    unused_variable = "this variable is never used"
    result=a+b
    return result

def format_string(text, capitalize=True):
  if capitalize:
      return text.upper()
  else:
    return text.lower()

class DataProcessor:
    def __init__(self,data):
        self.data=data
        self.processed_data=None
        
    def process(self):
        if self.data:
            self.processed_data=[item*2 for item in self.data if isinstance(item,int)]
        return self.processed_data
        
    def get_summary(self):
        if self.processed_data:
            return {"count":len(self.processed_data),"sum":sum(self.processed_data),"average":sum(self.processed_data)/len(self.processed_data)}
        else:
            return {}

def very_long_function_name_that_exceeds_typical_line_length_limits_and_should_trigger_linting_warnings(parameter_one, parameter_two, parameter_three, parameter_four):
    return parameter_one + parameter_two + parameter_three + parameter_four

def mixed_quotes_function():
    message1 = "This uses double quotes"
    message2 = 'This uses single quotes'
    message3 = """This uses triple double quotes"""
    message4 = '''This uses triple single quotes'''
    return message1+message2+message3+message4

# Missing blank lines before function definitions and inconsistent spacing

def function_with_bad_spacing():
    x=1
    y= 2
    z =3
    w = 4
    result=x+y+z+w
    return result

def test_calculate_sum():
    assert calculate_sum(2, 3) == 5
    assert calculate_sum(-1, 1) == 0
    assert calculate_sum(0, 0) == 0

def test_format_string():
    assert format_string("hello") == "HELLO"
    assert format_string("WORLD", False) == "world"

def test_data_processor():
    processor = DataProcessor([1, 2, 3, "text", 4])
    result = processor.process()
    assert result == [2, 4, 6, 8]
    summary = processor.get_summary()
    assert summary["count"] == 4
    assert summary["sum"] == 20