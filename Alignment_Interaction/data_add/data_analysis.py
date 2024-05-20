import json

file_path = 'case_analysis_processed.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# count the number of correct predictions
correct_count = 0
for item in data:
    if item['correct']:
        correct_count += 1
print(correct_count)
# print ratio
print(correct_count / len(data))

# count ratio of e1_has_img
hs_img_count = 0
for item in data:
    if item['e1_has_img']:
        hs_img_count += 1
print(hs_img_count / len(data))



# Initialize a counter for entities with images among incorrect predictions
hs_img_incorrect_count = 0
incorrect_count = 0

# Iterate through the data to count entities with images among incorrect predictions
for item in data:
    if not item['correct']:  # Focus on incorrect predictions
        incorrect_count += 1
        if item['e1_has_img']:  # Check if the entity has an image
            hs_img_incorrect_count += 1

# Calculate and print the ratio for entities with images among incorrect predictions
if incorrect_count > 0:  # Ensure there is at least one incorrect prediction to avoid division by zero
    print("Ratio of entities with images among incorrect predictions:", hs_img_incorrect_count / incorrect_count)
else:
    print("There are no incorrect predictions.")