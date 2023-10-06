from django.shortcuts import render

# Create your views here.
def front_view(request):
    context = {"x": 23}
    return render(request, "form.html", context)
