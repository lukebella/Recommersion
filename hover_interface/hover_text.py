class HoverText:
    def __init__(self, filepath):
        self.hover_dict = {}
        self.default = "Welcome to Recommersion! To get started, record your voice with \"Speak Emotion\" button above or adjust the valence and arousal sliders parameters on the right."

        with open(filepath, "r") as file:
            for line in file:
                key, value = line.strip().split("=", 1)
                self.hover_dict[key] = value

    def get_widget(self, widget_name):
        return self.hover_dict.get(widget_name, self.default)