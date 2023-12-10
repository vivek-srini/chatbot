# myapp/forms.py

from django import forms

class KeywordForm(forms.Form):
    file = forms.FileField(label='Upload a file')
    keyword = forms.CharField(label='Enter a keyword', max_length=100)

class TradeExclusionForm(forms.Form):
    file = forms.FileField(label='Upload a file')

class SignUpForm(forms.Form):
    username = forms.CharField(label='Enter a username', max_length=100)
    password = forms.CharField(label='Enter a password', max_length=100)

class LogInForm(forms.Form):
    username = forms.CharField(label='Enter a username', max_length=100)
    password = forms.CharField(label='Enter a password', max_length=100)

class LogOutForm(forms.Form):
    pass