generated_text = 'How far the horse has travelled in the twilight of midday after the french have drunk nothing but ale!'
pixel_values = []


for char in generated_text:
    if char in ' \n':
        pixel_values.append(0)  # Using 0 for spaces and newlines, adjust as needed
    else:
        pixel_values.append(ord(char))

print(pixel_values)
